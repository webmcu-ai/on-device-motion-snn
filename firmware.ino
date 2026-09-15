// ======================================================
// XIAO ML KIT (OR XIAO ESP32S3 SENSE)
// FULL MOTION / IMU ML — STREAMING SNN + SURROGATE GRADIENTS — v003
//
// Builds on v002. Two independent, coexisting classifiers:
//
//   ANN MODEL  (unchanged from v001/v002) — windowed Conv1D+Dense, trained
//   with ordinary backprop+Adam on 40-sample raw-accel windows. This is
//   your known-working 3-class baseline; use it for comparison.
//
//   STREAMING SNN MODEL (new in v003) — a genuinely different topology:
//     - Delta/level-crossing spike encoding (ON/OFF per axis), as in v002.
//     - A CAUSAL conv1 layer: a 5-sample ring buffer produces one LIF
//       output the instant a new IMU sample arrives, not a static
//       36-step block computed after a full window. This is what makes
//       continuous, non-windowed classification possible.
//     - Dense1/Dense2/Output are also true LIF neurons updated every
//       tick, fed by the causal conv's pooled output. Temporal context
//       comes from membrane leak/integration over time, NOT from
//       flattening a time window — which is why dense1 here is a
//       (CONV1_FILTERS -> DENSE1_SIZE) recurrent layer instead of the
//       ANN's (CONV1_FLAT -> DENSE1_SIZE) flattened-window layer. This
//       is a structurally different, smaller model, not a drop-in
//       channel-count swap of the ANN.
//     - TRAINED with real surrogate-gradient backprop-through-time
//       (BPTT): forward pass over a captured 40-tick repetition is
//       recorded tick-by-tick, then gradients are computed backward
//       through the LIF recurrences using a "fast sigmoid" surrogate
//       derivative in place of the (non-differentiable) spike step
//       function, with a DETACHED reset (standard practice — gradient
//       does not flow through the reset subtraction itself). This is
//       the same trick libraries like snnTorch/Norse use internally.
//     - INFERENCE runs continuously, tick by tick, off a persistent
//       streaming state — it never needs to wait for a full window.
//
// HONESTY / DEBUGGING NOTES (read before flashing):
//   - This BPTT implementation is written carefully but UNTESTED on
//     real hardware/data — expect to tune LEARNING_RATE, LIF_THRESHOLD,
//     LIF_LEAK, DELTA_THRESHOLD, and SNN_WEIGHT_SCALE. Serial prints
//     per-tick spike counts and the training loss so you can see
//     whether: (a) spikes are dying out entirely (lower LIF_THRESHOLD
//     or raise SNN_WEIGHT_SCALE), or (b) everything spikes every tick
//     (raise LIF_THRESHOLD / lower SNN_WEIGHT_SCALE / raise DELTA_THRESHOLD).
//   - The "Train+Infer" mode trains once per completed repetition, not
//     literally every tick — genuine continuous weight updates on every
//     sample forever is a further step once this baseline is validated.
//   - Layer freezing is implemented as real gates on the Adam update
//     calls, for both models, toggled via Serial single-key commands
//     (menu space on this OLED is too tight for 4 more items) — see the
//     "FREEZE COMMANDS" block near loop().
//
// By Jeremy Ellis
// With free tier assistance from: Claude (SNN/BPTT conversion), ChatGPT
//   (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!  MIT license
//
// lib_deps = olikraus/U8g2 @ ^2.35.30
//            Seeed Arduino LSM6DS3
// board_build.arduino.memory_type = qio_opi
//

#include <LSM6DS3.h>
#include <Wire.h>
#include "FS.h"
#include "SD.h"
#include "SPI.h"
#include <vector>
#include <algorithm>
#include <U8g2lib.h>

LSM6DS3 myIMU(I2C_MODE, 0x6A);
U8G2_SSD1306_72X40_ER_1_HW_I2C u8g2(U8G2_R2, U8X8_PIN_NONE);

// ======================================================
// CONFIGURATION
// ======================================================
#define NUM_CLASSES 3
String myClassLabels[NUM_CLASSES] = {"0Still", "1Punch", "2Wave"};

// 5 actions beyond the per-class collectors:
//   Train ANN, Infer ANN, Train SNN, Infer SNN (continuous), Train+Infer SNN
#define NUM_ACTIONS 5
const char* myActionLabels[NUM_ACTIONS] = {
  "Train ANN", "Infer ANN", "Train SNN", "Infer SNN", "Train+Infer SNN"
};
const int myTotalItems = NUM_CLASSES + NUM_ACTIONS;

float LEARNING_RATE      = 0.001f;
int   BATCH_SIZE         = 6;
int   TARGET_EPOCHS      = 30;
int   VALIDATION_SAMPLES = 3;

// ======================================================
// SHARED ARCHITECTURE CONSTANTS
// ======================================================
#define IMU_TIMESTEPS     40
#define IMU_AXES           3
#define SAMPLE_INTERVAL_MS 25
#define SNN_CHANNELS      (IMU_AXES * 2)   // ON/OFF per axis = 6

#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (IMU_TIMESTEPS - CONV1_KERNEL + 1)
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)     // ANN model only (window-flatten)

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16
#define DENSE2_WEIGHTS  (DENSE1_SIZE * DENSE2_SIZE)
#define OUTPUT_WEIGHTS  (DENSE2_SIZE * NUM_CLASSES)

// ANN model sizing (unchanged topology from v001/v002)
#define INPUT_SIZE_ANN     (IMU_TIMESTEPS * IMU_AXES)
#define CONV1_WEIGHTS_ANN  (CONV1_KERNEL * IMU_AXES * CONV1_FILTERS)
#define DENSE1_WEIGHTS_ANN (CONV1_FLAT * DENSE1_SIZE)

// Streaming SNN model sizing (per-tick recurrent, no window-flatten)
#define CONV1_WEIGHTS_SNN  (CONV1_KERNEL * SNN_CHANNELS * CONV1_FILTERS)
#define DENSE1_WEIGHTS_SNN (CONV1_FILTERS * DENSE1_SIZE)

// ======================================================
// SPIKE ENCODING
// ======================================================
float DELTA_THRESHOLD[IMU_AXES] = { 0.15f, 0.15f, 0.15f };

// ======================================================
// LIF + SURROGATE GRADIENT PARAMETERS
// ======================================================
float LIF_LEAK         = 0.90f;   // membrane decay per tick
float LIF_THRESHOLD    = 1.00f;   // spike threshold
float SNN_WEIGHT_SCALE = 1.00f;   // scales synaptic current
float SNN_TRACE_LEAK   = 0.95f;   // decay of the readout trace used for classification
#define SURROGATE_K 25.0f         // fast-sigmoid surrogate steepness

inline float mySurrogateDeriv(float memPre) {
  float x = memPre - LIF_THRESHOLD;
  float d = 1.0f + SURROGATE_K * fabsf(x);
  return 1.0f / (d * d);
}

// ======================================================
// NORMALIZATION (shared, from calibration)
// ======================================================
float myAccelMean[IMU_AXES] = { 0.0f, 0.0f, 1.0f };
float myAccelStd [IMU_AXES] = { 1.0f, 1.0f, 1.0f };
#define CALIB_SAMPLES 80

// ======================================================
// TOUCH INPUT (unchanged)
// ======================================================
const int myThresholdPress = 1100, myThresholdRelease = 900;
struct TouchState {
  bool isTouching = false; int tapCount = 0;
  unsigned long firstTapTime=0, lastReleaseTime=0, lastCheckTime=0;
  const unsigned long tapWindow=800, debounceDelay=50; const int longPressTaps=3;
};
TouchState myTouch;
unsigned long myLastActivityTime=0, myLastTapTime=0;
const int myTapCooldown = 250;
int  myMenuIndex = 1;
bool myIsSelected = false;
bool mySDavailable = false;

int myReadTouch() { int s=0; for(int i=0;i<3;i++){s+=analogRead(A0);delayMicroseconds(100);} return s/3; }
void myResetTouchState(){ myTouch.isTouching=false; myTouch.tapCount=0; myTouch.firstTapTime=0; myTouch.lastReleaseTime=0; myTouch.lastCheckTime=0; }
void myUpdateTouchState(){
  unsigned long now=millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool active = myTouch.isTouching ? (val>myThresholdRelease) : (val>myThresholdPress);
  if (active && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    if (myTouch.tapCount==0 || (now-myTouch.firstTapTime<myTouch.tapWindow)) { if(myTouch.tapCount==0) myTouch.firstTapTime=now; myTouch.tapCount++; }
    else { myTouch.tapCount=1; myTouch.firstTapTime=now; }
  }
  if (!active && myTouch.isTouching) { myTouch.isTouching=false; myTouch.lastReleaseTime=now; }
}
int myCheckTouchInput(){
  myUpdateTouchState();
  unsigned long now=millis();
  if (myTouch.tapCount>0 && !myTouch.isTouching) {
    if (now-myTouch.firstTapTime>myTouch.tapWindow) {
      int r=(myTouch.tapCount>=myTouch.longPressTaps)?2:1; myResetTouchState(); return r;
    }
  }
  return 0;
}
void myCheckTouchBackground(){ myUpdateTouchState(); }

// ======================================================
// UTILITY
// ======================================================
inline float myClip(float v,float mn=-100,float mx=100){ if(isnan(v)||isinf(v)) return 0; return constrain(v,mn,mx); }
inline float myLeakyRelu(float x){ return x>0?x:0.1f*x; }
inline float myLeakyReluDeriv(float x){ return x>0?1.0f:0.1f; }
void mySoftmax(float* x,int size){
  float mx=x[0]; for(int i=1;i<size;i++) if(x[i]>mx) mx=x[i];
  float sum=0; for(int i=0;i<size;i++){x[i]=exp(x[i]-mx); sum+=x[i];}
  for(int i=0;i<size;i++) x[i]/=sum;
}
void myNormalizeInput(float* buf){
  for(int t=0;t<IMU_TIMESTEPS;t++) for(int a=0;a<IMU_AXES;a++){
    int idx=t*IMU_AXES+a;
    buf[idx]=(buf[idx]-myAccelMean[a])/(myAccelStd[a]+1e-8f);
    buf[idx]=myClip(buf[idx],-5.0f,5.0f);
  }
}
// Delta/level-crossing spike encoder: normalized window -> IMU_TIMESTEPS x SNN_CHANNELS spikes
void myDeltaEncode(float* normalizedBuf, float* spikeOut){
  for (int c=0;c<SNN_CHANNELS;c++) spikeOut[c]=0.0f;
  for (int t=1;t<IMU_TIMESTEPS;t++){
    for (int a=0;a<IMU_AXES;a++){
      float d = normalizedBuf[t*IMU_AXES+a] - normalizedBuf[(t-1)*IMU_AXES+a];
      spikeOut[t*SNN_CHANNELS+a*2+0] = (d >  DELTA_THRESHOLD[a]) ? 1.0f : 0.0f;
      spikeOut[t*SNN_CHANNELS+a*2+1] = (d < -DELTA_THRESHOLD[a]) ? 1.0f : 0.0f;
    }
  }
}

// ======================================================
// CALIBRATION (unchanged)
// ======================================================
void myCalibrate(){
  if (mySDavailable && SD.exists("/header/myCalib.bin")) {
    File f=SD.open("/header/myCalib.bin",FILE_READ);
    if (f && f.size()==IMU_AXES*2*4) { f.read((uint8_t*)myAccelMean,IMU_AXES*4); f.read((uint8_t*)myAccelStd,IMU_AXES*4); f.close(); return; }
    if (f) f.close();
  }
  u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,10,"Calibrating...");u8g2.drawStr(0,22,"Keep still!");}while(u8g2.nextPage());
  delay(500);
  float sum[IMU_AXES]={0,0,0}, sum2[IMU_AXES]={0,0,0};
  for (int i=0;i<CALIB_SAMPLES;i++){
    float v[IMU_AXES]={myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ()};
    for(int a=0;a<IMU_AXES;a++){sum[a]+=v[a];sum2[a]+=v[a]*v[a];}
    delay(SAMPLE_INTERVAL_MS);
  }
  for(int a=0;a<IMU_AXES;a++){
    myAccelMean[a]=sum[a]/CALIB_SAMPLES;
    float var=(sum2[a]/CALIB_SAMPLES)-(myAccelMean[a]*myAccelMean[a]);
    myAccelStd[a]=max(sqrt(var),0.01f);
  }
  if (mySDavailable) {
    if (!SD.exists("/header")) SD.mkdir("/header");
    File f=SD.open("/header/myCalib.bin",FILE_WRITE);
    if (f){ f.write((uint8_t*)myAccelMean,IMU_AXES*4); f.write((uint8_t*)myAccelStd,IMU_AXES*4); f.close(); }
  }
}

// ======================================================
// ANN MODEL (unchanged topology/training from v001/v002 — your
// known-working 3-class baseline)
// ======================================================
struct AnnModel {
  bool trained=false; int adamStep=0;
  bool freezeConv1=false, freezeDense1=false, freezeDense2=false, freezeOutput=false;
  float *conv1_w,*conv1_b,*dense1_w,*dense1_b,*dense2_w,*dense2_b,*output_w,*output_b;
  float *conv1_w_grad,*conv1_b_grad,*dense1_w_grad,*dense1_b_grad,*dense2_w_grad,*dense2_b_grad,*output_w_grad,*output_b_grad;
  float *conv1_w_m,*conv1_w_v,*conv1_b_m,*conv1_b_v;
  float *dense1_w_m,*dense1_w_v,*dense1_b_m,*dense1_b_v;
  float *dense2_w_m,*dense2_w_v,*dense2_b_m,*dense2_b_v;
  float *output_w_m,*output_w_v,*output_b_m,*output_b_v;
};
AnnModel gAnn;

float* myAnnConv1Out=nullptr; float* myAnnPool1Out=nullptr;
float* myAnnDense1Out=nullptr; float* myAnnDense2Out=nullptr; float* myAnnFinal=nullptr;
float* myAnnOutDelta=nullptr; float* myAnnD2Delta=nullptr; float* myAnnD1Delta=nullptr;
float* myAnnPoolDelta=nullptr; float* myAnnConv1Delta=nullptr;
float* myAnnRawBuf=nullptr;   // IMU_TIMESTEPS*IMU_AXES, normalized in place

void myAdamUpdate(float* w,float* grad,float* mArr,float* vArr,int size,float lr,int step){
  const float b1=0.9f,b2=0.999f,eps=1e-8f;
  float bc1=1.0f-pow(b1,step), bc2=1.0f-pow(b2,step);
  for (int i=0;i<size;i++){
    mArr[i]=b1*mArr[i]+(1-b1)*grad[i];
    vArr[i]=b2*vArr[i]+(1-b2)*grad[i]*grad[i];
    w[i]-=lr*(mArr[i]/bc1)/(sqrt(vArr[i]/bc2)+eps);
  }
}

void myAllocateAnn(){
  gAnn.conv1_w=(float*)ps_malloc(CONV1_WEIGHTS_ANN*sizeof(float));  gAnn.conv1_b=(float*)ps_malloc(CONV1_FILTERS*sizeof(float));
  gAnn.dense1_w=(float*)ps_malloc(DENSE1_WEIGHTS_ANN*sizeof(float)); gAnn.dense1_b=(float*)ps_malloc(DENSE1_SIZE*sizeof(float));
  gAnn.dense2_w=(float*)ps_malloc(DENSE2_WEIGHTS*sizeof(float));     gAnn.dense2_b=(float*)ps_malloc(DENSE2_SIZE*sizeof(float));
  gAnn.output_w=(float*)ps_malloc(OUTPUT_WEIGHTS*sizeof(float));     gAnn.output_b=(float*)ps_malloc(NUM_CLASSES*sizeof(float));

  gAnn.conv1_w_grad=(float*)ps_malloc(CONV1_WEIGHTS_ANN*sizeof(float)); gAnn.conv1_b_grad=(float*)ps_malloc(CONV1_FILTERS*sizeof(float));
  gAnn.dense1_w_grad=(float*)ps_malloc(DENSE1_WEIGHTS_ANN*sizeof(float)); gAnn.dense1_b_grad=(float*)ps_malloc(DENSE1_SIZE*sizeof(float));
  gAnn.dense2_w_grad=(float*)ps_malloc(DENSE2_WEIGHTS*sizeof(float)); gAnn.dense2_b_grad=(float*)ps_malloc(DENSE2_SIZE*sizeof(float));
  gAnn.output_w_grad=(float*)ps_malloc(OUTPUT_WEIGHTS*sizeof(float)); gAnn.output_b_grad=(float*)ps_malloc(NUM_CLASSES*sizeof(float));

  gAnn.conv1_w_m=(float*)ps_calloc(CONV1_WEIGHTS_ANN,sizeof(float)); gAnn.conv1_w_v=(float*)ps_calloc(CONV1_WEIGHTS_ANN,sizeof(float));
  gAnn.conv1_b_m=(float*)ps_calloc(CONV1_FILTERS,sizeof(float));     gAnn.conv1_b_v=(float*)ps_calloc(CONV1_FILTERS,sizeof(float));
  gAnn.dense1_w_m=(float*)ps_calloc(DENSE1_WEIGHTS_ANN,sizeof(float)); gAnn.dense1_w_v=(float*)ps_calloc(DENSE1_WEIGHTS_ANN,sizeof(float));
  gAnn.dense1_b_m=(float*)ps_calloc(DENSE1_SIZE,sizeof(float));      gAnn.dense1_b_v=(float*)ps_calloc(DENSE1_SIZE,sizeof(float));
  gAnn.dense2_w_m=(float*)ps_calloc(DENSE2_WEIGHTS,sizeof(float));   gAnn.dense2_w_v=(float*)ps_calloc(DENSE2_WEIGHTS,sizeof(float));
  gAnn.dense2_b_m=(float*)ps_calloc(DENSE2_SIZE,sizeof(float));      gAnn.dense2_b_v=(float*)ps_calloc(DENSE2_SIZE,sizeof(float));
  gAnn.output_w_m=(float*)ps_calloc(OUTPUT_WEIGHTS,sizeof(float));   gAnn.output_w_v=(float*)ps_calloc(OUTPUT_WEIGHTS,sizeof(float));
  gAnn.output_b_m=(float*)ps_calloc(NUM_CLASSES,sizeof(float));      gAnn.output_b_v=(float*)ps_calloc(NUM_CLASSES,sizeof(float));

  myAnnConv1Out=(float*)ps_malloc(CONV1_OUT_STEPS*CONV1_FILTERS*sizeof(float));
  myAnnPool1Out=(float*)ps_malloc(CONV1_FLAT*sizeof(float));
  myAnnDense1Out=(float*)ps_malloc(DENSE1_SIZE*sizeof(float));
  myAnnDense2Out=(float*)ps_malloc(DENSE2_SIZE*sizeof(float));
  myAnnFinal=(float*)ps_malloc(NUM_CLASSES*sizeof(float));
  myAnnOutDelta=(float*)ps_malloc(NUM_CLASSES*sizeof(float));
  myAnnD2Delta=(float*)ps_malloc(DENSE2_SIZE*sizeof(float));
  myAnnD1Delta=(float*)ps_malloc(DENSE1_SIZE*sizeof(float));
  myAnnPoolDelta=(float*)ps_malloc(CONV1_FLAT*sizeof(float));
  myAnnConv1Delta=(float*)ps_malloc(CONV1_OUT_STEPS*CONV1_FILTERS*sizeof(float));
  myAnnRawBuf=(float*)ps_malloc(IMU_TIMESTEPS*IMU_AXES*sizeof(float));

  float c1=sqrt(2.0f/(CONV1_KERNEL*IMU_AXES));
  for(int i=0;i<CONV1_WEIGHTS_ANN;i++) gAnn.conv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1;
  for(int i=0;i<CONV1_FILTERS;i++) gAnn.conv1_b[i]=0;
  float d1=sqrt(2.0f/CONV1_FLAT);
  for(int i=0;i<DENSE1_WEIGHTS_ANN;i++) gAnn.dense1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d1;
  for(int i=0;i<DENSE1_SIZE;i++) gAnn.dense1_b[i]=0;
  float d2=sqrt(2.0f/DENSE1_SIZE);
  for(int i=0;i<DENSE2_WEIGHTS;i++) gAnn.dense2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d2;
  for(int i=0;i<DENSE2_SIZE;i++) gAnn.dense2_b[i]=0;
  float od=sqrt(2.0f/DENSE2_SIZE);
  for(int i=0;i<OUTPUT_WEIGHTS;i++) gAnn.output_w[i]=((float)rand()/RAND_MAX-0.5f)*2*od;
  for(int i=0;i<NUM_CLASSES;i++) gAnn.output_b[i]=0;
}

void myAnnForward(float* input){
  for(int s=0;s<CONV1_OUT_STEPS;s++) for(int f=0;f<CONV1_FILTERS;f++){
    float sum=gAnn.conv1_b[f];
    for(int k=0;k<CONV1_KERNEL;k++) for(int a=0;a<IMU_AXES;a++)
      sum+=input[(s+k)*IMU_AXES+a]*gAnn.conv1_w[(k*IMU_AXES+a)*CONV1_FILTERS+f];
    myAnnConv1Out[s*CONV1_FILTERS+f]=myLeakyRelu(sum);
  }
  for(int s=0;s<POOL1_STEPS;s++) for(int f=0;f<CONV1_FILTERS;f++){
    float a=myAnnConv1Out[(s*2)*CONV1_FILTERS+f], b=myAnnConv1Out[(s*2+1)*CONV1_FILTERS+f];
    myAnnPool1Out[s*CONV1_FILTERS+f]=max(a,b);
  }
  for(int j=0;j<DENSE1_SIZE;j++){ float sum=gAnn.dense1_b[j]; for(int i=0;i<CONV1_FLAT;i++) sum+=myAnnPool1Out[i]*gAnn.dense1_w[i*DENSE1_SIZE+j]; myAnnDense1Out[j]=myLeakyRelu(sum); }
  for(int j=0;j<DENSE2_SIZE;j++){ float sum=gAnn.dense2_b[j]; for(int i=0;i<DENSE1_SIZE;i++) sum+=myAnnDense1Out[i]*gAnn.dense2_w[i*DENSE2_SIZE+j]; myAnnDense2Out[j]=myLeakyRelu(sum); }
  for(int j=0;j<NUM_CLASSES;j++){ float sum=gAnn.output_b[j]; for(int i=0;i<DENSE2_SIZE;i++) sum+=myAnnDense2Out[i]*gAnn.output_w[i*NUM_CLASSES+j]; myAnnFinal[j]=sum; }
  mySoftmax(myAnnFinal,NUM_CLASSES);
}
void myAnnZeroGrad(){
  memset(gAnn.conv1_w_grad,0,CONV1_WEIGHTS_ANN*sizeof(float)); memset(gAnn.conv1_b_grad,0,CONV1_FILTERS*sizeof(float));
  memset(gAnn.dense1_w_grad,0,DENSE1_WEIGHTS_ANN*sizeof(float)); memset(gAnn.dense1_b_grad,0,DENSE1_SIZE*sizeof(float));
  memset(gAnn.dense2_w_grad,0,DENSE2_WEIGHTS*sizeof(float)); memset(gAnn.dense2_b_grad,0,DENSE2_SIZE*sizeof(float));
  memset(gAnn.output_w_grad,0,OUTPUT_WEIGHTS*sizeof(float)); memset(gAnn.output_b_grad,0,NUM_CLASSES*sizeof(float));
}
void myAnnBackward(float* input,int label){
  for(int j=0;j<NUM_CLASSES;j++) myAnnOutDelta[j]=myAnnFinal[j]-(j==label?1.0f:0.0f);
  for(int i=0;i<DENSE2_SIZE;i++) for(int j=0;j<NUM_CLASSES;j++) gAnn.output_w_grad[i*NUM_CLASSES+j]+=myAnnDense2Out[i]*myAnnOutDelta[j];
  for(int j=0;j<NUM_CLASSES;j++) gAnn.output_b_grad[j]+=myAnnOutDelta[j];
  for(int i=0;i<DENSE2_SIZE;i++){ float s=0; for(int j=0;j<NUM_CLASSES;j++) s+=gAnn.output_w[i*NUM_CLASSES+j]*myAnnOutDelta[j]; myAnnD2Delta[i]=s*myLeakyReluDeriv(myAnnDense2Out[i]); }
  for(int i=0;i<DENSE1_SIZE;i++) for(int j=0;j<DENSE2_SIZE;j++) gAnn.dense2_w_grad[i*DENSE2_SIZE+j]+=myAnnDense1Out[i]*myAnnD2Delta[j];
  for(int j=0;j<DENSE2_SIZE;j++) gAnn.dense2_b_grad[j]+=myAnnD2Delta[j];
  for(int i=0;i<DENSE1_SIZE;i++){ float s=0; for(int j=0;j<DENSE2_SIZE;j++) s+=gAnn.dense2_w[i*DENSE2_SIZE+j]*myAnnD2Delta[j]; myAnnD1Delta[i]=s*myLeakyReluDeriv(myAnnDense1Out[i]); }
  for(int i=0;i<CONV1_FLAT;i++) for(int j=0;j<DENSE1_SIZE;j++) gAnn.dense1_w_grad[i*DENSE1_SIZE+j]+=myAnnPool1Out[i]*myAnnD1Delta[j];
  for(int j=0;j<DENSE1_SIZE;j++) gAnn.dense1_b_grad[j]+=myAnnD1Delta[j];
  for(int s=0;s<POOL1_STEPS;s++) for(int f=0;f<CONV1_FILTERS;f++){
    float g=0; for(int j=0;j<DENSE1_SIZE;j++) g+=gAnn.dense1_w[(s*CONV1_FILTERS+f)*DENSE1_SIZE+j]*myAnnD1Delta[j];
    myAnnPoolDelta[s*CONV1_FILTERS+f]=g;
    float a=myAnnConv1Out[(s*2)*CONV1_FILTERS+f], b=myAnnConv1Out[(s*2+1)*CONV1_FILTERS+f];
    myAnnConv1Delta[(s*2)*CONV1_FILTERS+f]=(a>=b)?g:0.0f;
    myAnnConv1Delta[(s*2+1)*CONV1_FILTERS+f]=(b>a)?g:0.0f;
  }
  for(int s=0;s<CONV1_OUT_STEPS;s++) for(int f=0;f<CONV1_FILTERS;f++){
    float d=myAnnConv1Delta[s*CONV1_FILTERS+f]*myLeakyReluDeriv(myAnnConv1Out[s*CONV1_FILTERS+f]);
    gAnn.conv1_b_grad[f]+=d;
    for(int k=0;k<CONV1_KERNEL;k++) for(int a=0;a<IMU_AXES;a++)
      gAnn.conv1_w_grad[(k*IMU_AXES+a)*CONV1_FILTERS+f]+=input[(s+k)*IMU_AXES+a]*d;
  }
}
bool myAnnLoadSample(const char* path,float* outBuf){
  File f=SD.open(path); if(!f) return false;
  for(int t=0;t<IMU_TIMESTEPS;t++){
    for(int a=0;a<IMU_AXES;a++){ outBuf[t*IMU_AXES+a]=f.parseFloat(); if(a<IMU_AXES-1) while(f.available()&&f.peek()==',') f.read(); }
    while(f.available()&&(f.peek()=='\n'||f.peek()=='\r')) f.read();
  }
  f.close(); myNormalizeInput(outBuf); return true;
}
bool myAnnLoadWeights(){
  if(!mySDavailable||!SD.exists("/header/myAnnWeights.bin")) return false;
  File f=SD.open("/header/myAnnWeights.bin",FILE_READ); if(!f) return false;
  f.read((uint8_t*)gAnn.conv1_w,CONV1_WEIGHTS_ANN*4); f.read((uint8_t*)gAnn.conv1_b,CONV1_FILTERS*4);
  f.read((uint8_t*)gAnn.dense1_w,DENSE1_WEIGHTS_ANN*4); f.read((uint8_t*)gAnn.dense1_b,DENSE1_SIZE*4);
  f.read((uint8_t*)gAnn.dense2_w,DENSE2_WEIGHTS*4); f.read((uint8_t*)gAnn.dense2_b,DENSE2_SIZE*4);
  f.read((uint8_t*)gAnn.output_w,OUTPUT_WEIGHTS*4); f.read((uint8_t*)gAnn.output_b,NUM_CLASSES*4);
  f.close(); gAnn.trained=true; return true;
}
void myAnnSaveWeights(){
  if(!mySDavailable) return; if(!SD.exists("/header")) SD.mkdir("/header");
  File f=SD.open("/header/myAnnWeights.bin",FILE_WRITE);
  if(f){
    f.write((uint8_t*)gAnn.conv1_w,CONV1_WEIGHTS_ANN*4); f.write((uint8_t*)gAnn.conv1_b,CONV1_FILTERS*4);
    f.write((uint8_t*)gAnn.dense1_w,DENSE1_WEIGHTS_ANN*4); f.write((uint8_t*)gAnn.dense1_b,DENSE1_SIZE*4);
    f.write((uint8_t*)gAnn.dense2_w,DENSE2_WEIGHTS*4); f.write((uint8_t*)gAnn.dense2_b,DENSE2_SIZE*4);
    f.write((uint8_t*)gAnn.output_w,OUTPUT_WEIGHTS*4); f.write((uint8_t*)gAnn.output_b,NUM_CLASSES*4);
    f.close();
  }
}

// ======================================================
// SHARED: data collection + training-file bookkeeping
// (writes raw accel CSVs used by BOTH models)
// ======================================================
struct TrainingItem { String path; int label; };
std::vector<TrainingItem> myTrainingData;

int myCountSamples(int classIdx){
  if(!mySDavailable) return 0;
  String path="/motion/"+myClassLabels[classIdx];
  File root=SD.open(path); if(!root) return 0;
  int count=0; while(File f=root.openNextFile()){ if(!f.isDirectory()&&String(f.name()).endsWith(".csv")) count++; f.close(); }
  root.close(); return count;
}
bool myCaptureSample(int classIdx){
  String folderPath="/motion/"+myClassLabels[classIdx];
  if(!SD.exists("/motion")) SD.mkdir("/motion");
  if(!SD.exists(folderPath)) SD.mkdir(folderPath);
  int n=myCountSamples(classIdx);
  File f=SD.open(folderPath+"/s"+String(n)+".csv",FILE_WRITE);
  if(!f) return false;
  for(int t=0;t<IMU_TIMESTEPS;t++){
    unsigned long tS=millis();
    f.printf("%.5f,%.5f,%.5f\n",myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ());
    long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
  }
  f.close(); return true;
}
void myBuildTrainingList(int classCounts[NUM_CLASSES]){
  myTrainingData.clear();
  for(int c=0;c<NUM_CLASSES;c++){
    String path="/motion/"+myClassLabels[c];
    File root=SD.open(path); if(!root) continue;
    while(File file=root.openNextFile()){
      String name=file.name();
      if(!file.isDirectory()&&name.endsWith(".csv")){ myTrainingData.push_back({path+"/"+name,c}); classCounts[c]++; }
      file.close();
    }
    root.close();
  }
}
void myActionCollect(int classIdx){
  if(!mySDavailable){ myResetMenuState(); return; }
  myResetTouchState();
  int cc=myCountSamples(classIdx);
  u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,myClassLabels[classIdx].c_str());u8g2.drawStr(0,18,"TAP=Capture");u8g2.drawStr(0,28,"HOLD=Exit");char b[20];snprintf(b,20,"Count: %d",cc);u8g2.drawStr(0,38,b);}while(u8g2.nextPage());
  while(true){
    if(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L'){myResetMenuState();return;} if(c=='t'||c=='T'){delay(1000); if(myCaptureSample(classIdx)) cc++;} }
    int ta=myCheckTouchInput();
    if(ta==2){ myResetMenuState(); return; }
    if(ta==1){
      delay(1000);
      if(myCaptureSample(classIdx)){
        cc++;
        u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,myClassLabels[classIdx].c_str());char b[20];snprintf(b,20,"Saved: %d",cc);u8g2.drawStr(0,20,b);u8g2.drawStr(0,32,"TAP=More");}while(u8g2.nextPage());
      }
    }
  }
}

// ======================================================
// ANN TRAIN / INFER ACTIONS (freeze-aware)
// ======================================================
void myActionTrainAnn(){
  if(!mySDavailable){ myResetMenuState(); return; }
  int cc[NUM_CLASSES]={}; myBuildTrainingList(cc);
  std::random_shuffle(myTrainingData.begin(),myTrainingData.end());
  int valCount=0; std::vector<TrainingItem> valData;
  if(VALIDATION_SAMPLES>0){
    int held[NUM_CLASSES]={}; std::vector<TrainingItem> trainOnly;
    for(auto& it: myTrainingData){ if(held[it.label]<VALIDATION_SAMPLES){valData.push_back(it);held[it.label]++;valCount++;} else trainOnly.push_back(it); }
    myTrainingData=trainOnly;
  }
  for(int epoch=0;epoch<TARGET_EPOCHS;epoch++){
    std::random_shuffle(myTrainingData.begin(),myTrainingData.end());
    float loss=0; int correct=0, processed=0; myAnnZeroGrad();
    for(int si=0;si<(int)myTrainingData.size();si++){
      myCheckTouchBackground();
      if(!myAnnLoadSample(myTrainingData[si].path.c_str(),myAnnRawBuf)) continue;
      myAnnForward(myAnnRawBuf);
      int label=myTrainingData[si].label;
      loss+=-log(max(myAnnFinal[label],1e-7f));
      int pred=0; for(int j=1;j<NUM_CLASSES;j++) if(myAnnFinal[j]>myAnnFinal[pred]) pred=j;
      if(pred==label) correct++;
      myAnnBackward(myAnnRawBuf,label);
      processed++;
      if((si+1)%BATCH_SIZE==0 || si==(int)myTrainingData.size()-1){
        float sc=1.0f/processed;
        for(int k=0;k<CONV1_WEIGHTS_ANN;k++) gAnn.conv1_w_grad[k]*=sc;
        for(int k=0;k<CONV1_FILTERS;k++) gAnn.conv1_b_grad[k]*=sc;
        for(int k=0;k<DENSE1_WEIGHTS_ANN;k++) gAnn.dense1_w_grad[k]*=sc;
        for(int k=0;k<DENSE1_SIZE;k++) gAnn.dense1_b_grad[k]*=sc;
        for(int k=0;k<DENSE2_WEIGHTS;k++) gAnn.dense2_w_grad[k]*=sc;
        for(int k=0;k<DENSE2_SIZE;k++) gAnn.dense2_b_grad[k]*=sc;
        for(int k=0;k<OUTPUT_WEIGHTS;k++) gAnn.output_w_grad[k]*=sc;
        for(int k=0;k<NUM_CLASSES;k++) gAnn.output_b_grad[k]*=sc;
        gAnn.adamStep++;
        if(!gAnn.freezeConv1)  { myAdamUpdate(gAnn.conv1_w,gAnn.conv1_w_grad,gAnn.conv1_w_m,gAnn.conv1_w_v,CONV1_WEIGHTS_ANN,LEARNING_RATE,gAnn.adamStep);
                                  myAdamUpdate(gAnn.conv1_b,gAnn.conv1_b_grad,gAnn.conv1_b_m,gAnn.conv1_b_v,CONV1_FILTERS,LEARNING_RATE,gAnn.adamStep); }
        if(!gAnn.freezeDense1) { myAdamUpdate(gAnn.dense1_w,gAnn.dense1_w_grad,gAnn.dense1_w_m,gAnn.dense1_w_v,DENSE1_WEIGHTS_ANN,LEARNING_RATE,gAnn.adamStep);
                                  myAdamUpdate(gAnn.dense1_b,gAnn.dense1_b_grad,gAnn.dense1_b_m,gAnn.dense1_b_v,DENSE1_SIZE,LEARNING_RATE,gAnn.adamStep); }
        if(!gAnn.freezeDense2) { myAdamUpdate(gAnn.dense2_w,gAnn.dense2_w_grad,gAnn.dense2_w_m,gAnn.dense2_w_v,DENSE2_WEIGHTS,LEARNING_RATE,gAnn.adamStep);
                                  myAdamUpdate(gAnn.dense2_b,gAnn.dense2_b_grad,gAnn.dense2_b_m,gAnn.dense2_b_v,DENSE2_SIZE,LEARNING_RATE,gAnn.adamStep); }
        if(!gAnn.freezeOutput) { myAdamUpdate(gAnn.output_w,gAnn.output_w_grad,gAnn.output_w_m,gAnn.output_w_v,OUTPUT_WEIGHTS,LEARNING_RATE,gAnn.adamStep);
                                  myAdamUpdate(gAnn.output_b,gAnn.output_b_grad,gAnn.output_b_m,gAnn.output_b_v,NUM_CLASSES,LEARNING_RATE,gAnn.adamStep); }
        myAnnZeroGrad(); processed=0;
      }
    }
    float valAcc=0;
    if(valCount>0){ int vc=0; for(auto& vi: valData){ if(!myAnnLoadSample(vi.path.c_str(),myAnnRawBuf)) continue; myAnnForward(myAnnRawBuf); int p=0; for(int j=1;j<NUM_CLASSES;j++) if(myAnnFinal[j]>myAnnFinal[p]) p=j; if(p==vi.label) vc++; } valAcc=100.0f*vc/valCount; }
    Serial.printf("[ANN] Epoch %d/%d Loss=%.4f TrainAcc=%.1f%% ValAcc=%.1f%%\n",epoch+1,TARGET_EPOCHS,loss/max((int)myTrainingData.size(),1),100.0f*correct/max((int)myTrainingData.size(),1),valAcc);
  }
  gAnn.trained=true; myAnnSaveWeights();
  u8g2.firstPage(); do{u8g2.drawStr(0,15,"ANN trained");}while(u8g2.nextPage()); delay(1200); myResetMenuState();
}
void myActionInferAnn(){
  if(!gAnn.trained){ u8g2.firstPage(); do{u8g2.drawStr(0,15,"No ANN weights");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  int windowCount=0, voteBuf[3]={0,0,0}, voteIdx=0, finalPred=0;
  while(true){
    if(myCheckTouchInput()==2){ myResetMenuState(); return; }
    if(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L'){myResetMenuState();return;} }
    for(int t=0;t<IMU_TIMESTEPS;t++){
      unsigned long tS=millis();
      myAnnRawBuf[t*IMU_AXES+0]=myIMU.readFloatAccelX(); myAnnRawBuf[t*IMU_AXES+1]=myIMU.readFloatAccelY(); myAnnRawBuf[t*IMU_AXES+2]=myIMU.readFloatAccelZ();
      long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
    }
    myNormalizeInput(myAnnRawBuf); myAnnForward(myAnnRawBuf);
    int rawPred=0; for(int j=1;j<NUM_CLASSES;j++) if(myAnnFinal[j]>myAnnFinal[rawPred]) rawPred=j;
    windowCount++; voteBuf[voteIdx%3]=rawPred; voteIdx++;
    int votes[NUM_CLASSES]={}; for(int v=0;v<3;v++) votes[voteBuf[v]]++;
    finalPred=0; for(int j=1;j<NUM_CLASSES;j++) if(votes[j]>votes[finalPred]) finalPred=j;
    Serial.printf("[ANN] #%d raw=%s vote=%s\n",windowCount,myClassLabels[rawPred].c_str(),myClassLabels[finalPred].c_str());
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"ANN:");u8g2.drawStr(0,18,myClassLabels[finalPred].c_str());}while(u8g2.nextPage());
  }
}

// ======================================================
// STREAMING SNN MODEL — surrogate-gradient trained, causal, continuous
// ======================================================
struct SnnModel {
  bool trained=false; int adamStep=0;
  bool freezeConv1=false, freezeDense1=false, freezeDense2=false, freezeOutput=false;
  float *conv1_w,*conv1_b,*dense1_w,*dense1_b,*dense2_w,*dense2_b,*output_w,*output_b;
  float *conv1_w_grad,*conv1_b_grad,*dense1_w_grad,*dense1_b_grad,*dense2_w_grad,*dense2_b_grad,*output_w_grad,*output_b_grad;
  float *conv1_w_m,*conv1_w_v,*conv1_b_m,*conv1_b_v;
  float *dense1_w_m,*dense1_w_v,*dense1_b_m,*dense1_b_v;
  float *dense2_w_m,*dense2_w_v,*dense2_b_m,*dense2_b_v;
  float *output_w_m,*output_w_v,*output_b_m,*output_b_v;
};
SnnModel gSnn;

// Per-tick recorded state, for BPTT over one captured repetition (T=IMU_TIMESTEPS)
float* rConv1MemPre;   // T*CONV1_FILTERS
uint8_t* rConv1Spike;  // T*CONV1_FILTERS
float* rPooled;        // T*CONV1_FILTERS  (held pooled feature, 0/1 as float)
float* rDense1MemPre;  // T*DENSE1_SIZE
uint8_t* rDense1Spike; // T*DENSE1_SIZE
float* rDense2MemPre;  // T*DENSE2_SIZE
uint8_t* rDense2Spike; // T*DENSE2_SIZE
float* rOutCurrent;    // T*NUM_CLASSES
float* rTrace;         // T*NUM_CLASSES
float* mySnnSpikeInput; // T*SNN_CHANNELS, filled by myDeltaEncode before training

// Persistent state for continuous/streaming inference (independent of training)
float gStreamRing[CONV1_KERNEL][SNN_CHANNELS];
int   gStreamRingFill = 0;   // how many real samples are in the ring so far (for causal zero-pad)
int   gStreamTickParity = 0; // 0/1, tracks pooling pairs
float gStreamConv1MemPost[CONV1_FILTERS];
float gStreamPooledHeld[CONV1_FILTERS];
float gStreamDense1MemPost[DENSE1_SIZE];
float gStreamDense2MemPost[DENSE2_SIZE];
float gStreamTrace[NUM_CLASSES];
long  gStreamConv1Spikes=0, gStreamDense1Spikes=0, gStreamDense2Spikes=0, gStreamTicks=0; // diagnostics

void mySnnAllocate(){
  gSnn.conv1_w=(float*)ps_malloc(CONV1_WEIGHTS_SNN*sizeof(float));  gSnn.conv1_b=(float*)ps_calloc(CONV1_FILTERS,sizeof(float));
  gSnn.dense1_w=(float*)ps_malloc(DENSE1_WEIGHTS_SNN*sizeof(float)); gSnn.dense1_b=(float*)ps_calloc(DENSE1_SIZE,sizeof(float));
  gSnn.dense2_w=(float*)ps_malloc(DENSE2_WEIGHTS*sizeof(float));     gSnn.dense2_b=(float*)ps_calloc(DENSE2_SIZE,sizeof(float));
  gSnn.output_w=(float*)ps_malloc(OUTPUT_WEIGHTS*sizeof(float));     gSnn.output_b=(float*)ps_calloc(NUM_CLASSES,sizeof(float));

  gSnn.conv1_w_grad=(float*)ps_malloc(CONV1_WEIGHTS_SNN*sizeof(float)); gSnn.conv1_b_grad=(float*)ps_malloc(CONV1_FILTERS*sizeof(float));
  gSnn.dense1_w_grad=(float*)ps_malloc(DENSE1_WEIGHTS_SNN*sizeof(float)); gSnn.dense1_b_grad=(float*)ps_malloc(DENSE1_SIZE*sizeof(float));
  gSnn.dense2_w_grad=(float*)ps_malloc(DENSE2_WEIGHTS*sizeof(float)); gSnn.dense2_b_grad=(float*)ps_malloc(DENSE2_SIZE*sizeof(float));
  gSnn.output_w_grad=(float*)ps_malloc(OUTPUT_WEIGHTS*sizeof(float)); gSnn.output_b_grad=(float*)ps_malloc(NUM_CLASSES*sizeof(float));

  gSnn.conv1_w_m=(float*)ps_calloc(CONV1_WEIGHTS_SNN,sizeof(float)); gSnn.conv1_w_v=(float*)ps_calloc(CONV1_WEIGHTS_SNN,sizeof(float));
  gSnn.conv1_b_m=(float*)ps_calloc(CONV1_FILTERS,sizeof(float));     gSnn.conv1_b_v=(float*)ps_calloc(CONV1_FILTERS,sizeof(float));
  gSnn.dense1_w_m=(float*)ps_calloc(DENSE1_WEIGHTS_SNN,sizeof(float)); gSnn.dense1_w_v=(float*)ps_calloc(DENSE1_WEIGHTS_SNN,sizeof(float));
  gSnn.dense1_b_m=(float*)ps_calloc(DENSE1_SIZE,sizeof(float));      gSnn.dense1_b_v=(float*)ps_calloc(DENSE1_SIZE,sizeof(float));
  gSnn.dense2_w_m=(float*)ps_calloc(DENSE2_WEIGHTS,sizeof(float));   gSnn.dense2_w_v=(float*)ps_calloc(DENSE2_WEIGHTS,sizeof(float));
  gSnn.dense2_b_m=(float*)ps_calloc(DENSE2_SIZE,sizeof(float));      gSnn.dense2_b_v=(float*)ps_calloc(DENSE2_SIZE,sizeof(float));
  gSnn.output_w_m=(float*)ps_calloc(OUTPUT_WEIGHTS,sizeof(float));   gSnn.output_w_v=(float*)ps_calloc(OUTPUT_WEIGHTS,sizeof(float));
  gSnn.output_b_m=(float*)ps_calloc(NUM_CLASSES,sizeof(float));      gSnn.output_b_v=(float*)ps_calloc(NUM_CLASSES,sizeof(float));

  rConv1MemPre=(float*)ps_malloc(IMU_TIMESTEPS*CONV1_FILTERS*sizeof(float));
  rConv1Spike=(uint8_t*)ps_malloc(IMU_TIMESTEPS*CONV1_FILTERS*sizeof(uint8_t));
  rPooled=(float*)ps_malloc(IMU_TIMESTEPS*CONV1_FILTERS*sizeof(float));
  rDense1MemPre=(float*)ps_malloc(IMU_TIMESTEPS*DENSE1_SIZE*sizeof(float));
  rDense1Spike=(uint8_t*)ps_malloc(IMU_TIMESTEPS*DENSE1_SIZE*sizeof(uint8_t));
  rDense2MemPre=(float*)ps_malloc(IMU_TIMESTEPS*DENSE2_SIZE*sizeof(float));
  rDense2Spike=(uint8_t*)ps_malloc(IMU_TIMESTEPS*DENSE2_SIZE*sizeof(uint8_t));
  rOutCurrent=(float*)ps_malloc(IMU_TIMESTEPS*NUM_CLASSES*sizeof(float));
  rTrace=(float*)ps_malloc(IMU_TIMESTEPS*NUM_CLASSES*sizeof(float));
  mySnnSpikeInput=(float*)ps_malloc(IMU_TIMESTEPS*SNN_CHANNELS*sizeof(float));

  // He-ish init, scaled down a bit since spiking currents accumulate over many ticks
  float c1=sqrt(2.0f/(CONV1_KERNEL*SNN_CHANNELS));
  for(int i=0;i<CONV1_WEIGHTS_SNN;i++) gSnn.conv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1*0.5f;
  float d1=sqrt(2.0f/CONV1_FILTERS);
  for(int i=0;i<DENSE1_WEIGHTS_SNN;i++) gSnn.dense1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d1*0.5f;
  float d2=sqrt(2.0f/DENSE1_SIZE);
  for(int i=0;i<DENSE2_WEIGHTS;i++) gSnn.dense2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d2*0.5f;
  float od=sqrt(2.0f/DENSE2_SIZE);
  for(int i=0;i<OUTPUT_WEIGHTS;i++) gSnn.output_w[i]=((float)rand()/RAND_MAX-0.5f)*2*od;
}

void mySnnResetStreamState(){
  memset(gStreamRing,0,sizeof(gStreamRing));
  gStreamRingFill=0; gStreamTickParity=0;
  memset(gStreamConv1MemPost,0,sizeof(gStreamConv1MemPost));
  memset(gStreamPooledHeld,0,sizeof(gStreamPooledHeld));
  memset(gStreamDense1MemPost,0,sizeof(gStreamDense1MemPost));
  memset(gStreamDense2MemPost,0,sizeof(gStreamDense2MemPost));
  memset(gStreamTrace,0,sizeof(gStreamTrace));
  gStreamConv1Spikes=gStreamDense1Spikes=gStreamDense2Spikes=gStreamTicks=0;
}

uint8_t gConvSpikePrevTick[CONV1_FILTERS];   // holds one tick's conv spikes while waiting for its pool partner

// One tick of the streaming SNN. rawSpikeSample = SNN_CHANNELS floats (0/1) for THIS instant.
// If record==true, also writes into the r* arrays at index tIdx for later BPTT.
// Returns the current argmax(trace) prediction.
int mySnnTick(float* rawSpikeSample, bool record, int tIdx){
  for(int k=0;k<CONV1_KERNEL-1;k++) memcpy(gStreamRing[k],gStreamRing[k+1],SNN_CHANNELS*sizeof(float));
  memcpy(gStreamRing[CONV1_KERNEL-1],rawSpikeSample,SNN_CHANNELS*sizeof(float));
  if (gStreamRingFill < CONV1_KERNEL) gStreamRingFill++;
  gStreamTicks++;

  // ---- causal conv1: ring[0..K-1] = last K samples, oldest..newest, zero-padded until filled ----
  uint8_t curConvSpike[CONV1_FILTERS];
  for(int f=0;f<CONV1_FILTERS;f++){
    float current = gSnn.conv1_b[f] * SNN_WEIGHT_SCALE;
    int missing = CONV1_KERNEL - gStreamRingFill;
    for(int k=0;k<CONV1_KERNEL;k++){
      if (k < missing) continue;
      for(int a=0;a<SNN_CHANNELS;a++)
        current += gStreamRing[k][a] * gSnn.conv1_w[(k*SNN_CHANNELS+a)*CONV1_FILTERS+f] * SNN_WEIGHT_SCALE;
    }
    float memPre = gStreamConv1MemPost[f]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rConv1MemPre[tIdx*CONV1_FILTERS+f]=memPre; rConv1Spike[tIdx*CONV1_FILTERS+f]=spike; }
    if (spike) gStreamConv1Spikes++;
    gStreamConv1MemPost[f] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    curConvSpike[f] = spike;
  }

  // ---- pooling: OR pairs of conv ticks; the pooled feature holds for 2 ticks ----
  // parity 0 = first tick of a pair (stash it), parity 1 = second tick (compute + hold the OR)
  if (gStreamTickParity == 0) {
    memcpy(gConvSpikePrevTick, curConvSpike, sizeof(curConvSpike));
  } else {
    for (int f=0; f<CONV1_FILTERS; f++)
      gStreamPooledHeld[f] = (gConvSpikePrevTick[f] | curConvSpike[f]) ? 1.0f : 0.0f;
  }
  gStreamTickParity = 1 - gStreamTickParity;
  if (record) for (int f=0; f<CONV1_FILTERS; f++) rPooled[tIdx*CONV1_FILTERS+f] = gStreamPooledHeld[f];

  // ---- dense1 (LIF, per-tick recurrent — this replaces the ANN's window-flatten) ----
  uint8_t d1spike[DENSE1_SIZE];
  for(int j=0;j<DENSE1_SIZE;j++){
    float current = gSnn.dense1_b[j] * SNN_WEIGHT_SCALE;
    for(int i=0;i<CONV1_FILTERS;i++)
      current += gStreamPooledHeld[i] * gSnn.dense1_w[i*DENSE1_SIZE+j] * SNN_WEIGHT_SCALE;
    float memPre = gStreamDense1MemPost[j]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rDense1MemPre[tIdx*DENSE1_SIZE+j]=memPre; rDense1Spike[tIdx*DENSE1_SIZE+j]=spike; }
    if (spike) gStreamDense1Spikes++;
    gStreamDense1MemPost[j] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    d1spike[j]=spike;
  }

  // ---- dense2 (LIF) ----
  uint8_t d2spike[DENSE2_SIZE];
  for(int j=0;j<DENSE2_SIZE;j++){
    float current = gSnn.dense2_b[j] * SNN_WEIGHT_SCALE;
    for(int i=0;i<DENSE1_SIZE;i++)
      current += d1spike[i] * gSnn.dense2_w[i*DENSE2_SIZE+j] * SNN_WEIGHT_SCALE;
    float memPre = gStreamDense2MemPost[j]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rDense2MemPre[tIdx*DENSE2_SIZE+j]=memPre; rDense2Spike[tIdx*DENSE2_SIZE+j]=spike; }
    if (spike) gStreamDense2Spikes++;
    gStreamDense2MemPost[j] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    d2spike[j]=spike;
  }

  // ---- output: current-based readout (no spiking here), fed into a decaying classification trace ----
  for(int c=0;c<NUM_CLASSES;c++){
    float current = gSnn.output_b[c];
    for(int i=0;i<DENSE2_SIZE;i++) current += d2spike[i] * gSnn.output_w[i*NUM_CLASSES+c];
    gStreamTrace[c] = gStreamTrace[c]*SNN_TRACE_LEAK + current;
    if (record){ rOutCurrent[tIdx*NUM_CLASSES+c]=current; rTrace[tIdx*NUM_CLASSES+c]=gStreamTrace[c]; }
  }

  int pred=0; for(int c=1;c<NUM_CLASSES;c++) if(gStreamTrace[c]>gStreamTrace[pred]) pred=c;
  return pred;
}

// ======================================================
// SURROGATE-GRADIENT BACKPROP-THROUGH-TIME
// Runs after a recorded 40-tick forward pass (mySnnTick called with
// record=true for t=0..IMU_TIMESTEPS-1). Uses the fast-sigmoid surrogate
// derivative in place of the spike step function's true derivative, and
// a DETACHED reset (standard practice: gradient does not flow back
// through the "subtract threshold on spike" term itself).
// Accumulates into gSnn.*_grad; caller must zero grads first and apply
// an Adam step after.
// ======================================================
void mySnnZeroGrad(){
  memset(gSnn.conv1_w_grad,0,CONV1_WEIGHTS_SNN*sizeof(float));   memset(gSnn.conv1_b_grad,0,CONV1_FILTERS*sizeof(float));
  memset(gSnn.dense1_w_grad,0,DENSE1_WEIGHTS_SNN*sizeof(float)); memset(gSnn.dense1_b_grad,0,DENSE1_SIZE*sizeof(float));
  memset(gSnn.dense2_w_grad,0,DENSE2_WEIGHTS*sizeof(float));     memset(gSnn.dense2_b_grad,0,DENSE2_SIZE*sizeof(float));
  memset(gSnn.output_w_grad,0,OUTPUT_WEIGHTS*sizeof(float));     memset(gSnn.output_b_grad,0,NUM_CLASSES*sizeof(float));
}

void mySnnBackwardBPTT(int label){
  // Loss: softmax cross-entropy on the FINAL tick's decaying trace, treated as logits.
  float probs[NUM_CLASSES];
  for(int c=0;c<NUM_CLASSES;c++) probs[c]=rTrace[(IMU_TIMESTEPS-1)*NUM_CLASSES+c];
  mySoftmax(probs,NUM_CLASSES);
  float dTrace[NUM_CLASSES];
  for(int c=0;c<NUM_CLASSES;c++) dTrace[c]=probs[c]-(c==label?1.0f:0.0f);

  float dDense2MemPost[DENSE2_SIZE]; memset(dDense2MemPost,0,sizeof(dDense2MemPost));
  float dDense1MemPost[DENSE1_SIZE]; memset(dDense1MemPost,0,sizeof(dDense1MemPost));
  float dConv1MemPost[CONV1_FILTERS]; memset(dConv1MemPost,0,sizeof(dConv1MemPost));
  static float dPooledAccum[IMU_TIMESTEPS][CONV1_FILTERS];
  memset(dPooledAccum,0,sizeof(dPooledAccum));

  // ---- walk backward through dense2/dense1 and the output/trace readout ----
  for (int t = IMU_TIMESTEPS-1; t >= 0; t--) {
    float dOutCurrent[NUM_CLASSES];
    for(int c=0;c<NUM_CLASSES;c++) dOutCurrent[c]=dTrace[c];
    for(int c=0;c<NUM_CLASSES;c++) dTrace[c]*=SNN_TRACE_LEAK;   // propagate trace recurrence to t-1

    uint8_t* d2spike_t = &rDense2Spike[t*DENSE2_SIZE];
    float dDense2Spike[DENSE2_SIZE]; memset(dDense2Spike,0,sizeof(dDense2Spike));
    for(int c=0;c<NUM_CLASSES;c++){
      if(!gSnn.freezeOutput){
        gSnn.output_b_grad[c]+=dOutCurrent[c];
        for(int i=0;i<DENSE2_SIZE;i++) gSnn.output_w_grad[i*NUM_CLASSES+c]+=d2spike_t[i]*dOutCurrent[c];
      }
      for(int i=0;i<DENSE2_SIZE;i++) dDense2Spike[i]+=gSnn.output_w[i*NUM_CLASSES+c]*dOutCurrent[c];
    }

    float dDense2MemPre[DENSE2_SIZE];
    for(int j=0;j<DENSE2_SIZE;j++){
      float sg = mySurrogateDeriv(rDense2MemPre[t*DENSE2_SIZE+j]);
      dDense2MemPre[j] = dDense2Spike[j]*sg + dDense2MemPost[j];   // detached-reset: dMemPost/dMemPre = 1
    }
    uint8_t* d1spike_t = &rDense1Spike[t*DENSE1_SIZE];
    float dDense1Spike[DENSE1_SIZE]; memset(dDense1Spike,0,sizeof(dDense1Spike));
    for(int j=0;j<DENSE2_SIZE;j++){
      if(!gSnn.freezeDense2){
        gSnn.dense2_b_grad[j]+=dDense2MemPre[j];
        for(int i=0;i<DENSE1_SIZE;i++) gSnn.dense2_w_grad[i*DENSE2_SIZE+j]+=d1spike_t[i]*dDense2MemPre[j];
      }
      for(int i=0;i<DENSE1_SIZE;i++) dDense1Spike[i]+=gSnn.dense2_w[i*DENSE2_SIZE+j]*dDense2MemPre[j];
      dDense2MemPost[j] = dDense2MemPre[j]*LIF_LEAK;
    }

    float dDense1MemPre[DENSE1_SIZE];
    for(int j=0;j<DENSE1_SIZE;j++){
      float sg = mySurrogateDeriv(rDense1MemPre[t*DENSE1_SIZE+j]);
      dDense1MemPre[j] = dDense1Spike[j]*sg + dDense1MemPost[j];
    }
    for(int j=0;j<DENSE1_SIZE;j++){
      if(!gSnn.freezeDense1){
        gSnn.dense1_b_grad[j]+=dDense1MemPre[j];
        for(int i=0;i<CONV1_FILTERS;i++) gSnn.dense1_w_grad[i*DENSE1_SIZE+j]+=rPooled[t*CONV1_FILTERS+i]*dDense1MemPre[j];
      }
      for(int i=0;i<CONV1_FILTERS;i++) dPooledAccum[t][i]+=gSnn.dense1_w[i*DENSE1_SIZE+j]*dDense1MemPre[j];
      dDense1MemPost[j]=dDense1MemPre[j]*LIF_LEAK;
    }
  }

  // ---- route pooled-feature gradients back to the pair of conv1 ticks that produced them ----
  // (OR-pooling isn't differentiable; this routes the gradient to both source ticks equally —
  //  a straight-through approximation, same spirit as max-pool gradient routing.)
  static float dConvSpikeFromPool[IMU_TIMESTEPS][CONV1_FILTERS];
  memset(dConvSpikeFromPool,0,sizeof(dConvSpikeFromPool));
  for(int t=1;t<IMU_TIMESTEPS;t++){
    int pairStart = 2*((t-1)/2);
    for(int f=0;f<CONV1_FILTERS;f++){
      dConvSpikeFromPool[pairStart][f]   += dPooledAccum[t][f];
      dConvSpikeFromPool[pairStart+1][f] += dPooledAccum[t][f];
    }
  }

  // ---- conv1 backward through time ----
  for (int t = IMU_TIMESTEPS-1; t>=0; t--){
    float dConv1MemPre[CONV1_FILTERS];
    for(int f=0;f<CONV1_FILTERS;f++){
      float sg = mySurrogateDeriv(rConv1MemPre[t*CONV1_FILTERS+f]);
      dConv1MemPre[f] = dConvSpikeFromPool[t][f]*sg + dConv1MemPost[f];
    }
    if(!gSnn.freezeConv1){
      for(int f=0;f<CONV1_FILTERS;f++){
        gSnn.conv1_b_grad[f]+=dConv1MemPre[f];
        for(int k=0;k<CONV1_KERNEL;k++){
          int srcT = t-(CONV1_KERNEL-1)+k;
          if (srcT<0) continue;   // causal zero-padding region, no input to differentiate
          for(int a=0;a<SNN_CHANNELS;a++)
            gSnn.conv1_w_grad[(k*SNN_CHANNELS+a)*CONV1_FILTERS+f]+=mySnnSpikeInput[srcT*SNN_CHANNELS+a]*dConv1MemPre[f];
        }
      }
    }
    for(int f=0;f<CONV1_FILTERS;f++) dConv1MemPost[f]=dConv1MemPre[f]*LIF_LEAK;
  }
}

void mySnnApplyAdam(){
  gSnn.adamStep++;
  if(!gSnn.freezeConv1){
    myAdamUpdate(gSnn.conv1_w,gSnn.conv1_w_grad,gSnn.conv1_w_m,gSnn.conv1_w_v,CONV1_WEIGHTS_SNN,LEARNING_RATE,gSnn.adamStep);
    myAdamUpdate(gSnn.conv1_b,gSnn.conv1_b_grad,gSnn.conv1_b_m,gSnn.conv1_b_v,CONV1_FILTERS,LEARNING_RATE,gSnn.adamStep);
  }
  if(!gSnn.freezeDense1){
    myAdamUpdate(gSnn.dense1_w,gSnn.dense1_w_grad,gSnn.dense1_w_m,gSnn.dense1_w_v,DENSE1_WEIGHTS_SNN,LEARNING_RATE,gSnn.adamStep);
    myAdamUpdate(gSnn.dense1_b,gSnn.dense1_b_grad,gSnn.dense1_b_m,gSnn.dense1_b_v,DENSE1_SIZE,LEARNING_RATE,gSnn.adamStep);
  }
  if(!gSnn.freezeDense2){
    myAdamUpdate(gSnn.dense2_w,gSnn.dense2_w_grad,gSnn.dense2_w_m,gSnn.dense2_w_v,DENSE2_WEIGHTS,LEARNING_RATE,gSnn.adamStep);
    myAdamUpdate(gSnn.dense2_b,gSnn.dense2_b_grad,gSnn.dense2_b_m,gSnn.dense2_b_v,DENSE2_SIZE,LEARNING_RATE,gSnn.adamStep);
  }
  if(!gSnn.freezeOutput){
    myAdamUpdate(gSnn.output_w,gSnn.output_w_grad,gSnn.output_w_m,gSnn.output_w_v,OUTPUT_WEIGHTS,LEARNING_RATE,gSnn.adamStep);
    myAdamUpdate(gSnn.output_b,gSnn.output_b_grad,gSnn.output_b_m,gSnn.output_b_v,NUM_CLASSES,LEARNING_RATE,gSnn.adamStep);
  }
}

bool mySnnLoadWeights(){
  if(!mySDavailable||!SD.exists("/header/mySnnWeights.bin")) return false;
  File f=SD.open("/header/mySnnWeights.bin",FILE_READ); if(!f) return false;
  f.read((uint8_t*)gSnn.conv1_w,CONV1_WEIGHTS_SNN*4);   f.read((uint8_t*)gSnn.conv1_b,CONV1_FILTERS*4);
  f.read((uint8_t*)gSnn.dense1_w,DENSE1_WEIGHTS_SNN*4); f.read((uint8_t*)gSnn.dense1_b,DENSE1_SIZE*4);
  f.read((uint8_t*)gSnn.dense2_w,DENSE2_WEIGHTS*4);     f.read((uint8_t*)gSnn.dense2_b,DENSE2_SIZE*4);
  f.read((uint8_t*)gSnn.output_w,OUTPUT_WEIGHTS*4);     f.read((uint8_t*)gSnn.output_b,NUM_CLASSES*4);
  f.close(); gSnn.trained=true; return true;
}
void mySnnSaveWeights(){
  if(!mySDavailable) return; if(!SD.exists("/header")) SD.mkdir("/header");
  File f=SD.open("/header/mySnnWeights.bin",FILE_WRITE);
  if(f){
    f.write((uint8_t*)gSnn.conv1_w,CONV1_WEIGHTS_SNN*4);   f.write((uint8_t*)gSnn.conv1_b,CONV1_FILTERS*4);
    f.write((uint8_t*)gSnn.dense1_w,DENSE1_WEIGHTS_SNN*4); f.write((uint8_t*)gSnn.dense1_b,DENSE1_SIZE*4);
    f.write((uint8_t*)gSnn.dense2_w,DENSE2_WEIGHTS*4);     f.write((uint8_t*)gSnn.dense2_b,DENSE2_SIZE*4);
    f.write((uint8_t*)gSnn.output_w,OUTPUT_WEIGHTS*4);     f.write((uint8_t*)gSnn.output_b,NUM_CLASSES*4);
    f.close();
  }
}

float* mySnnRawBuf = nullptr;   // IMU_TIMESTEPS*IMU_AXES, scratch for loading CSVs for "Train SNN"
int myLastSelectedClass = 0;    // set whenever a class-collect menu item is visited; used by Train+Infer SNN

bool mySnnLoadSampleEncoded(const char* path){
  File f=SD.open(path); if(!f) return false;
  for(int t=0;t<IMU_TIMESTEPS;t++){
    for(int a=0;a<IMU_AXES;a++){ mySnnRawBuf[t*IMU_AXES+a]=f.parseFloat(); if(a<IMU_AXES-1) while(f.available()&&f.peek()==',') f.read(); }
    while(f.available()&&(f.peek()=='\n'||f.peek()=='\r')) f.read();
  }
  f.close();
  myNormalizeInput(mySnnRawBuf);
  myDeltaEncode(mySnnRawBuf, mySnnSpikeInput);
  return true;
}

// ======================================================
// TRAIN SNN — surrogate-gradient BPTT on previously-collected .csv reps
// (same data your ANN model trains from; one Adam step per repetition)
// ======================================================
void myActionTrainSnn(){
  if(!mySDavailable){ myResetMenuState(); return; }
  int cc[NUM_CLASSES]={}; myBuildTrainingList(cc);
  std::random_shuffle(myTrainingData.begin(),myTrainingData.end());
  int valCount=0; std::vector<TrainingItem> valData;
  if(VALIDATION_SAMPLES>0){
    int held[NUM_CLASSES]={}; std::vector<TrainingItem> trainOnly;
    for(auto& it: myTrainingData){ if(held[it.label]<VALIDATION_SAMPLES){valData.push_back(it);held[it.label]++;valCount++;} else trainOnly.push_back(it); }
    myTrainingData=trainOnly;
  }
  for(int epoch=0; epoch<TARGET_EPOCHS; epoch++){
    std::random_shuffle(myTrainingData.begin(),myTrainingData.end());
    float lossSum=0; int correct=0;
    for(size_t si=0; si<myTrainingData.size(); si++){
      myCheckTouchBackground();
      if(!mySnnLoadSampleEncoded(myTrainingData[si].path.c_str())) continue;
      mySnnResetStreamState();
      int label = myTrainingData[si].label;
      int pred=0;
      for(int t=0;t<IMU_TIMESTEPS;t++)
        pred = mySnnTick(&mySnnSpikeInput[t*SNN_CHANNELS], true, t);

      float probs[NUM_CLASSES];
      for(int c=0;c<NUM_CLASSES;c++) probs[c]=rTrace[(IMU_TIMESTEPS-1)*NUM_CLASSES+c];
      mySoftmax(probs,NUM_CLASSES);
      lossSum += -log(max(probs[label],1e-7f));
      if (pred==label) correct++;

      mySnnZeroGrad();
      mySnnBackwardBPTT(label);
      mySnnApplyAdam();
    }
    float valAcc=0;
    if(valCount>0){
      int vc=0;
      for(auto& vi: valData){
        if(!mySnnLoadSampleEncoded(vi.path.c_str())) continue;
        mySnnResetStreamState();
        int pred=0;
        for(int t=0;t<IMU_TIMESTEPS;t++) pred=mySnnTick(&mySnnSpikeInput[t*SNN_CHANNELS], false, 0);
        if (pred==vi.label) vc++;
      }
      valAcc=100.0f*vc/valCount;
    }
    Serial.printf("[SNN] Epoch %d/%d Loss=%.4f TrainAcc=%.1f%% ValAcc=%.1f%% (last-sample spikes: c1=%ld d1=%ld d2=%ld)\n",
      epoch+1,TARGET_EPOCHS, lossSum/max((int)myTrainingData.size(),1),
      100.0f*correct/max((int)myTrainingData.size(),1), valAcc,
      gStreamConv1Spikes, gStreamDense1Spikes, gStreamDense2Spikes);
    u8g2.firstPage();
    do{ u8g2.setFont(u8g2_font_5x7_tf); char b[24];
        snprintf(b,24,"SNN Ep %d/%d",epoch+1,TARGET_EPOCHS); u8g2.drawStr(0,8,b);
        snprintf(b,24,"Tr%.0f Val%.0f",100.0f*correct/max((int)myTrainingData.size(),1),valAcc); u8g2.drawStr(0,20,b);
    } while(u8g2.nextPage());
  }
  gSnn.trained=true; mySnnSaveWeights();
  u8g2.firstPage(); do{u8g2.drawStr(0,15,"SNN trained");}while(u8g2.nextPage()); delay(1200); myResetMenuState();
}

// ======================================================
// INFER SNN — continuous streaming classification, no windowing.
// Reads the trace every tick and reports it; never waits for 40 samples.
// ======================================================
void myActionInferSnnContinuous(){
  if(!gSnn.trained){ u8g2.firstPage(); do{u8g2.drawStr(0,15,"No SNN weights");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  mySnnResetStreamState();
  float prevNorm[IMU_AXES]={0,0,0}; bool havePrev=false;
  unsigned long tickCount=0;
  while(true){
    if(myCheckTouchInput()==2){ myResetMenuState(); return; }
    if(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L'){myResetMenuState();return;} }

    unsigned long tS=millis();
    float raw[IMU_AXES]={myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ()};
    float norm[IMU_AXES];
    for(int a=0;a<IMU_AXES;a++) norm[a]=myClip((raw[a]-myAccelMean[a])/(myAccelStd[a]+1e-8f),-5.0f,5.0f);

    float spikeTick[SNN_CHANNELS];
    for(int a=0;a<IMU_AXES;a++){
      float d = havePrev ? (norm[a]-prevNorm[a]) : 0.0f;
      spikeTick[a*2+0]=(d> DELTA_THRESHOLD[a])?1.0f:0.0f;
      spikeTick[a*2+1]=(d< -DELTA_THRESHOLD[a])?1.0f:0.0f;
    }
    memcpy(prevNorm,norm,sizeof(norm)); havePrev=true;

    int pred = mySnnTick(spikeTick, false, 0);
    tickCount++;

    if (tickCount % 8 == 0) {   // ~every 200ms, to avoid flooding Serial/OLED
      Serial.printf("[SNN-live] tick=%lu pred=%s\n", tickCount, myClassLabels[pred].c_str());
      u8g2.firstPage();
      do{ u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0,8,"SNN live:"); u8g2.drawStr(0,18,myClassLabels[pred].c_str());
          char b[20]; snprintf(b,20,"t=%lu",tickCount); u8g2.drawStr(0,28,b); } while(u8g2.nextPage());
    }
    long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
  }
}

// ======================================================
// TRAIN+INFER SNN — combined continuous mode. Streams live predictions
// WHILE capturing a repetition (labeled as whichever class you last
// visited in the menu), then runs one BPTT update from that same
// repetition the instant it ends.
// ======================================================
void myActionTrainInferSnn(){
  Serial.printf("[SNN Train+Infer] Labeling reps as '%s' (visit a class item first to change). TAP=start rep, HOLD=exit.\n",
                myClassLabels[myLastSelectedClass].c_str());
  myResetTouchState();
  while(true){
    int ta = myCheckTouchInput();
    if (ta==2) { myResetMenuState(); return; }
    bool startRep = (ta==1);
    if (!startRep && Serial.available()){
      char c=Serial.read();
      if(c=='l'||c=='L'){ myResetMenuState(); return; }
      if(c=='t'||c=='T') startRep=true;
    }
    if (!startRep) { delay(20); continue; }

    mySnnResetStreamState();
    float prevNorm[IMU_AXES]={0,0,0}; bool havePrev=false;
    int label = myLastSelectedClass;
    int lastPred=0;
    for(int t=0;t<IMU_TIMESTEPS;t++){
      unsigned long tS=millis();
      float raw[IMU_AXES]={myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ()};
      float norm[IMU_AXES];
      for(int a=0;a<IMU_AXES;a++) norm[a]=myClip((raw[a]-myAccelMean[a])/(myAccelStd[a]+1e-8f),-5.0f,5.0f);
      float spikeTick[SNN_CHANNELS];
      for(int a=0;a<IMU_AXES;a++){
        float d = havePrev ? (norm[a]-prevNorm[a]) : 0.0f;
        spikeTick[a*2+0]=(d>DELTA_THRESHOLD[a])?1.0f:0.0f;
        spikeTick[a*2+1]=(d<-DELTA_THRESHOLD[a])?1.0f:0.0f;
      }
      memcpy(prevNorm,norm,sizeof(norm)); havePrev=true;
      memcpy(&mySnnSpikeInput[t*SNN_CHANNELS], spikeTick, SNN_CHANNELS*sizeof(float));

      lastPred = mySnnTick(spikeTick, true, t);
      if (t%5==0) Serial.printf("  live tick %d -> %s\n", t, myClassLabels[lastPred].c_str());

      long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
    }

    mySnnZeroGrad();
    mySnnBackwardBPTT(label);
    mySnnApplyAdam();
    gSnn.trained = true;
    mySnnSaveWeights();

    Serial.printf("[SNN Train+Infer] Rep done. Labeled=%s LivePredAtEnd=%s\n",
                  myClassLabels[label].c_str(), myClassLabels[lastPred].c_str());
    u8g2.firstPage();
    do{ u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0,8,"Trained on:"); u8g2.drawStr(0,18,myClassLabels[label].c_str());
        u8g2.drawStr(0,28, lastPred==label ? "Pred: OK" : "Pred: MISS"); } while(u8g2.nextPage());
    delay(800);
  }
}

// ======================================================
// LAYER-FREEZE COMMANDS (Serial monitor)
//   A/S/D/F = toggle SNN freeze on conv1/dense1/dense2/output
//   a/s/d/f = toggle ANN freeze on conv1/dense1/dense2/output
// Use these before "Train SNN" / "Train ANN" / "Train+Infer SNN" to
// test whether freezing early layers reduces interference when adding
// new classes.
// ======================================================
void myHandleFreezeCommand(char c){
  switch(c){
    case 'A': gSnn.freezeConv1=!gSnn.freezeConv1;   Serial.printf("SNN conv1 frozen=%d\n",gSnn.freezeConv1); break;
    case 'S': gSnn.freezeDense1=!gSnn.freezeDense1; Serial.printf("SNN dense1 frozen=%d\n",gSnn.freezeDense1); break;
    case 'D': gSnn.freezeDense2=!gSnn.freezeDense2; Serial.printf("SNN dense2 frozen=%d\n",gSnn.freezeDense2); break;
    case 'F': gSnn.freezeOutput=!gSnn.freezeOutput; Serial.printf("SNN output frozen=%d\n",gSnn.freezeOutput); break;
    case 'a': gAnn.freezeConv1=!gAnn.freezeConv1;   Serial.printf("ANN conv1 frozen=%d\n",gAnn.freezeConv1); break;
    case 's': gAnn.freezeDense1=!gAnn.freezeDense1; Serial.printf("ANN dense1 frozen=%d\n",gAnn.freezeDense1); break;
    case 'd': gAnn.freezeDense2=!gAnn.freezeDense2; Serial.printf("ANN dense2 frozen=%d\n",gAnn.freezeDense2); break;
    case 'f': gAnn.freezeOutput=!gAnn.freezeOutput; Serial.printf("ANN output frozen=%d\n",gAnn.freezeOutput); break;
  }
}

// ======================================================
// MENU
// ======================================================
void myResetMenuState(){ myIsSelected=false; myResetTouchState(); myLastActivityTime=millis(); myDrawMenu(); }
String myMenuLabel(int idx){ if(idx<=NUM_CLASSES) return myClassLabels[idx-1]; return String(myActionLabels[idx-NUM_CLASSES-1]); }
void myDrawMenu(){
  Serial.println("\n=== MENU ===");
  for(int i=1;i<=myTotalItems;i++) Serial.printf("%s%d. %s\n",(i==myMenuIndex)?" > ":"   ",i,myMenuLabel(i).c_str());
  Serial.println("Freeze cmds: A/S/D/F=SNN conv1/d1/d2/out  a/s/d/f=ANN conv1/d1/d2/out");
  u8g2.firstPage();
  do{
    u8g2.setFont(u8g2_font_6x10_tf);
    u8g2.drawStr(0,8,"TAP:Next HOLD:Ok");
    int start=max(1,myMenuIndex-1);
    for(int i=0;i<3;i++){ int cur=start+i; if(cur>myTotalItems) break; int y=18+i*9; String line=(cur==myMenuIndex?"> ":"  ")+myMenuLabel(cur); u8g2.drawStr(0,y,line.c_str()); }
  } while(u8g2.nextPage());
}
void myExecuteMenuItem(int idx){
  if(idx<=NUM_CLASSES){ myLastSelectedClass = idx-1; myActionCollect(idx-1); return; }
  switch(idx-NUM_CLASSES-1){
    case 0: myActionTrainAnn(); break;
    case 1: myActionInferAnn(); break;
    case 2: myActionTrainSnn(); break;
    case 3: myActionInferSnnContinuous(); break;
    case 4: myActionTrainInferSnn(); break;
  }
}
void myHandleMenuNavigation(){
  unsigned long now=millis();
  if(!myIsSelected && Serial.available()){
    char c=Serial.read();
    if (c=='A'||c=='S'||c=='D'||c=='F'||c=='a'||c=='s'||c=='d'||c=='f'){ myHandleFreezeCommand(c); return; }
    if(c>='1'&&c<='9'){ int ni=c-'0'; if(ni<=myTotalItems){ myMenuIndex=ni; myIsSelected=true; myExecuteMenuItem(myMenuIndex);} }
    else if(c=='t'||c=='T'){ if(now-myLastTapTime>myTapCooldown){ myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1; myDrawMenu(); myLastTapTime=now; } }
    else if(c=='l'||c=='L'){ myIsSelected=true; myExecuteMenuItem(myMenuIndex); }
  }
  if(!myIsSelected){
    int ta=myCheckTouchInput();
    if(ta==1){ if(now-myLastTapTime>myTapCooldown){ myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1; myDrawMenu(); myLastTapTime=now; } }
    else if(ta==2){ myIsSelected=true; myExecuteMenuItem(myMenuIndex); }
  }
}

// ======================================================
// SETUP / LOOP
// ======================================================
void setup(){
  Serial.begin(115200);
  while(!Serial && millis()<3000);
  delay(1000);
  Serial.println("\n=== XIAO ESP32-S3 Motion ML — Streaming SNN + Surrogate Gradients v003 ===");
  Serial.println("Freeze toggles: A/S/D/F = SNN conv1/dense1/dense2/output, a/s/d/f = ANN equivalents");

  pinMode(A0, INPUT);
  u8g2.begin();

  pinMode(21, OUTPUT); digitalWrite(21, HIGH); delay(100);
  SPI.begin(); SPI.setFrequency(400000);
  mySDavailable = SD.begin(21, SPI, 400000, "/sd", 5, false);
  if(!mySDavailable){ SD.end(); Serial.println("No SD card - continuing without it"); }

  if (myIMU.begin()!=0){ Serial.println("ERROR: IMU init failed!"); while(1) delay(1000); }
  myCalibrate();

  myAllocateAnn();
  mySnnAllocate();
  mySnnRawBuf = (float*)ps_malloc(IMU_TIMESTEPS*IMU_AXES*sizeof(float));
  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  myAnnLoadWeights();
  mySnnLoadWeights();

  myResetMenuState();
  Serial.println("System ready.");
}

void loop(){
  myHandleMenuNavigation();
}
