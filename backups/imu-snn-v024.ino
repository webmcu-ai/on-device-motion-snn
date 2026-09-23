// ======================================================================
// XIAO ESP32S3 SENSE / XIAO ML KIT - MOTION "AIR-WRITING" WITH A SPIKING NEURAL NETWORK
// imu-snn-v024
//
// WHAT IT DOES
//   Move the board through the air to write letters. The board learns
//   your gestures, recognises them, and builds a line of text on the
//   Serial Monitor and the OLED.
//     Still  -> adds a space
//     Delete -> removes the last character (can repeat about once a second)
//     other  -> adds that class's letter
//
// WHAT'S NEW IN v024
//   Inference is no longer a fixed sliding window over the last WINDOW_TICKS
//   ticks. See GESTURE SEGMENTATION below and myGestureTick() /
//   myClassifyGesture() in the code. The SNN still predicts every tick (for
//   the live Serial trail), but nothing is typed until a complete gesture -
//   movement, then stillness - has been seen. "Still" (space) no longer has
//   to be recognised mid-gesture by the SNN at all: since it is, by
//   definition, an absence of movement, it is now detected as sustained
//   stillness while otherwise idle (STILL_HOLD_TICKS). The ANN path,
//   training code, spike encoding and SNN layers/BPTT are all unchanged.
//
// TWO CLASSIFIERS (both trained on-device from the same recorded gestures)
//   ANN  Windowed Conv1D + Dense network, ordinary backprop + Adam.
//        Looks at a whole 1-second window at once.
//   SNN  Streaming spiking network (the main topic of this lesson).
//        Processes one IMU sample at a time and never waits for a window.
//
// HOW THE SNN WORKS (signal flow)
//   1. ENCODE   Accelerometer -> spikes. Each axis has an ON and an OFF
//               channel that fire when the reading has changed by more
//               than SPIKE_DELTA_G since its last spike. A 7th "idle"
//               channel fires on ticks where nothing else did, so being
//               still is a real signal instead of silence.
//   2. CONV1    Causal 1D conv over the last 5 ticks -> 8 LIF neurons.
//   3. POOL     Every 2 ticks, OR the conv spikes together.
//   4. DENSE1   32 LIF neurons.
//   5. DENSE2   16 LIF neurons.
//   6. OUTPUT   One "trace" per class: a leaky running sum of the
//               output-layer current. Highest trace = current guess.
//   LIF neuron: the membrane voltage leaks a little each tick, adds up
//   incoming current, and fires (then drops by the threshold) when it
//   crosses LIF_THRESHOLD.
//
// HOW THE SNN LEARNS
//   Surrogate-gradient backprop-through-time (BPTT): the spike is a hard
//   step, so backward we pretend it was a smooth curve. Each recorded
//   1-second repetition gives one training example; the loss is measured
//   over the last part of the repetition so the network learns to HOLD
//   the right answer, not just to be right on the very last tick.
//
// HOW TEXT IS PRODUCED (the "output engine")
//   The SNN's per-class traces update every tick, but a class guess alone
//   never types anything - see GESTURE SEGMENTATION below. The one
//   exception is "Still": since it has no movement to segment, a separate,
//   much simpler timeout (STILL_HOLD_TICKS) types a space once the board
//   has sat still for a while.
//
// GESTURE SEGMENTATION (turns a whole air-written trajectory into one letter)
//   Typing straight from the streaming trace is unreliable: an unfinished W
//   can briefly look like an O, and a fixed-length window can commit before
//   the letter is even finished. So the output engine waits for a whole
//   gesture before deciding anything:
//     IDLE      No movement in progress. Watches for GESTURE_ONSET_TICKS of
//               movement in a row (a letter starting), or STILL_HOLD_TICKS
//               of stillness (types a space).
//     ACTIVE    The gesture is under way. The SNN keeps predicting every
//               tick (printed as "[GESTURE] ACTIVE" when verbose), but none
//               of it is typed yet. Evidence is accumulated for the whole
//               gesture (a running average of every tick's probabilities)
//               and, separately, for just its most recent FINAL_PORTION_TICKS
//               ticks, since a letter is only really identifiable once it is
//               complete.
//     ENDING    Movement has stopped; if it stays stopped for
//               GESTURE_END_STILL_TICKS the gesture is considered finished.
//               A brief pause mid-letter (shorter than that) just returns to
//               ACTIVE rather than ending the gesture early.
//     CLASSIFY  One decision, made from the accumulated evidence by blending
//               the whole-gesture average with the final-portion average
//               (GESTURE_FINAL_WEIGHT). A gesture that was too short
//               (MIN_GESTURE_TICKS) or not confident enough
//               (GESTURE_COMMIT_MIN_PROB / GESTURE_COMMIT_MARGIN) types
//               nothing.
//     COOLDOWN  A short settle (RESET_SETTLE_TICKS) while the just-reset SNN
//               dynamics stabilise, then back to IDLE to wait for the next
//               gesture.
//   Exactly one commit can come out of one gesture, so a long or repeated
//   motion can no longer "double-type" the way a fixed sliding window could.
//   Gesture boundaries come from movement -> stillness, not from a fixed
//   timeout, so MAX_GESTURE_TICKS only exists as a safety cap in case
//   stillness is never detected.
//
// USING IT (Serial Monitor at 115200; '$' reprints the menu)
//   MAIN MENU
//   letters    select a class to collect data for (case-insensitive)
//   ! / @      select the Still / Delete class (no letter of their own)
//   1-7        Train ANN / Infer ANN / Train SNN / Infer SNN / Train+Infer SNN /
//              AutoTrain ANN / AutoTrain SNN (6 and 7 are continuous, saving after every loop)
//   #          toggle verbose (debug) or quiet (transcript only) inference
//   ?          status report (scans the SD card, so it is slow)
//   % ^ & *    freeze SNN conv1 / dense1 / dense2 / output
//   ( ) - =    freeze ANN conv1 / dense1 / dense2 / output
//   , / .      next item / select item (or tap / 3x tap on the touch pad)
//   INSIDE A CLASS (collecting)
//   t          capture one 1-second gesture (or tap the touch pad)
//   a          AUTO-COLLECT 10 gestures in a row
//   Enter      (blank) repeats the last command
//   l          exit / back to menu        3x tap also exits
//   WHILE AUTO-TRAINING
//   l or 3x tap  stop (the loop in progress is discarded, earlier loops are saved)
//
// TUNING (all in the CONFIGURATION section)
//   Spikes dead or all-on?          LIF_THRESHOLD, SPIKE_DELTA_G
//   Letters typed too easily?       GESTURE_COMMIT_MIN_PROB, GESTURE_COMMIT_MARGIN (raise them)
//   Gesture ends too soon/too late? GESTURE_END_STILL_TICKS (raise if letters get cut off
//                                   mid-stroke, lower if there's a long pause before each letter)
//   Short flicks typed as letters?  MIN_GESTURE_TICKS (raise it)
//   Spaces too slow/fast to appear? STILL_HOLD_TICKS
//
// By Jeremy Ellis
// With free tier assistance from: Claude (SNN/BPTT conversion, gesture segmentation), ChatGPT
//   (Critique), Gemini (Research) and Copilot (Alternate)
// Use at your own risk!  MIT license
//
// lib_deps = olikraus/U8g2 @ ^2.35.30
//            Seeed Arduino LSM6DS3
// board_build.arduino.memory_type = qio_opi
// ======================================================================

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

// ======================================================================
// CONFIGURATION
// ======================================================================

// ---- Classes -----------------------------------------------------------
// Each class has three independent fields:
//   key    the hotkey. A letter is matched case-insensitively (w or W both
//          work); a symbol (like '!' or '@') is matched exactly, for the
//          two classes that have no natural letter of their own.
//   label  the text typed into the transcript and shown in the menu/OLED.
//   folder the SD card folder name under /motion/. Kept identical to the
//          numbered names used before v022 so samples collected on earlier
//          firmware still load without renaming anything on the card.
// To add a class, add one row below with a key that isn't reserved (see
// the list under the array) - NUM_CLASSES and everything sized from it
// updates automatically. myValidateClassKeys() (called once in setup())
// prints a Serial warning if two classes ever end up sharing a hotkey.
//   Reserved hotkeys - do not reuse these as a class's `key`:
//     Actions:  1 2 3 4 5 6 7
//     Freeze:   % ^ & * ( ) - =
//     Other:    # $ ? , .
struct ClassDef { char key; const char* label; const char* folder; };
const ClassDef myClasses[] = {
  { '!', "Still",  "0Still"  },
  { '@', "Delete", "1Delete" },
  { 'W', "W",      "2W"      },
  { 'O', "O",      "3O"      },
  { 'R', "R",      "4R"      },
  { 'D', "D",      "5D"      },
  { 'S', "S",      "6S"      },
  { 'A', "A",      "7A"      },
  { 'T', "T",      "8T"      },
  { 'E', "E",      "9E"      },
};
#define NUM_CLASSES ((int)(sizeof(myClasses)/sizeof(myClasses[0])))

#define NUM_ACTIONS 7
const char* myActionLabels[NUM_ACTIONS] = {
  "Train ANN", "Infer ANN", "Train SNN", "Infer SNN", "Train+Infer SNN",
  "AutoTrain ANN", "AutoTrain SNN"
};
const char myActionKeys[NUM_ACTIONS] = { '1','2','3','4','5','6','7' };
const int myTotalItems = NUM_CLASSES + NUM_ACTIONS;

// ---- Training (both models) ----------------------------------------------
float LEARNING_RATE      = 0.002f;
int   BATCH_SIZE         = 6;      // gestures per weight update
int   TARGET_EPOCHS      = 40;     // epochs per loop (a loop = one save when auto-training)
int   VALIDATION_SAMPLES = 3;      // gestures per class held out for testing

// ---- SNN: spike encoding and neurons -------------------------------------
const float SPIKE_DELTA_G = 0.03f; // accel change (in g) that makes a spike; sensor noise is ~0.005 g
float LIF_LEAK        = 0.90f;     // membrane voltage kept each tick
float LIF_THRESHOLD   = 0.50f;     // voltage at which a neuron fires
float SNN_TRACE_LEAK  = 0.95f;     // memory of the class readout trace
#define SURROGATE_K     10.0f      // smoothness of the surrogate gradient

// ---- SNN: training details -----------------------------------------------
const int   TRAIN_LOSS_FROM_TICK = 16;   // loss is measured on ticks 16..39 of each gesture
const int   AUG_MAX_SHIFT        = 4;    // random start-time shift (ticks) used while training
const float AUG_GAIN_RANGE       = 0.15f;// random +/- gesture size change while training

// ---- Output engine: SNN gesture segmentation ------------------------------
// The SNN predicts every tick, but nothing is typed until a complete gesture
// (movement, then stillness) has been observed. See myGestureTick() and
// myClassifyGesture() below, and the GESTURE SEGMENTATION notes up top.
const int   GESTURE_ONSET_TICKS     = 3;    // consecutive moving ticks before a gesture is considered STARTED
const int   GESTURE_END_STILL_TICKS = 8;    // consecutive still ticks before a gesture is considered FINISHED (~200ms)
const int   MIN_GESTURE_TICKS       = 12;   // gestures shorter than this are ignored as noise, not a letter
const int   MAX_GESTURE_TICKS       = 120;  // safety cap (~3s): forces classification even if stillness never comes
const int   FINAL_PORTION_TICKS     = 10;   // how many of the most recent ticks count as the gesture's "final portion"
const float GESTURE_FINAL_WEIGHT    = 0.5f; // blend weight: 0 = whole-gesture average only, 1 = final portion only
const int   RESET_SETTLE_TICKS      = 3;    // ticks ignored right after a commit while the SNN reset settles
const float GESTURE_COMMIT_MIN_PROB = 0.45f;// the whole-gesture leader's blended score must be at least this
const float GESTURE_COMMIT_MARGIN   = 0.15f;// ...and beat the runner-up by this much
const int   STILL_HOLD_TICKS        = 40;   // ticks of continued stillness (after any letter) before a space is typed
const unsigned long DELETE_REPEAT_MS = 800; // Delete may fire again after this long
const float ANN_COMMIT_MIN_PROB = 0.50f;   // ANN: a window is typed only if its top-class probability is at least this (0 = always type)
const int   ANN_CONFIRM_WINDOWS = 1;       // ANN: same answer this many 1-second windows in a row before typing (1 = every window; each typed letter restarts the count)

// ======================================================================
// ARCHITECTURE CONSTANTS
// ======================================================================
#define IMU_TIMESTEPS      40           // 40 samples x 25 ms = 1 second per gesture
#define IMU_AXES            3
#define SAMPLE_INTERVAL_MS 25
#define SNN_CHANNELS       (IMU_AXES * 2 + 1)   // ON/OFF per axis + 1 idle channel
#define SNN_IDLE_CH        (IMU_AXES * 2)

#define CONV1_KERNEL    5
#define CONV1_FILTERS   8
#define CONV1_OUT_STEPS (IMU_TIMESTEPS - CONV1_KERNEL + 1)
#define POOL1_STEPS     (CONV1_OUT_STEPS / 2)
#define CONV1_FLAT      (POOL1_STEPS * CONV1_FILTERS)   // ANN only

#define DENSE1_SIZE   32
#define DENSE2_SIZE   16
#define DENSE2_WEIGHTS  (DENSE1_SIZE * DENSE2_SIZE)
#define OUTPUT_WEIGHTS  (DENSE2_SIZE * NUM_CLASSES)

#define CONV1_WEIGHTS_ANN  (CONV1_KERNEL * IMU_AXES * CONV1_FILTERS)
#define DENSE1_WEIGHTS_ANN (CONV1_FLAT * DENSE1_SIZE)
#define CONV1_WEIGHTS_SNN  (CONV1_KERNEL * SNN_CHANNELS * CONV1_FILTERS)
#define DENSE1_WEIGHTS_SNN (CONV1_FILTERS * DENSE1_SIZE)

#define CALIB_SAMPLES 80

// ======================================================================
// SHARED STATE
// ======================================================================
float myAccelMean[IMU_AXES] = { 0.0f, 0.0f, 1.0f };   // ANN input normalisation (from calibration)
float myAccelStd [IMU_AXES] = { 1.0f, 1.0f, 1.0f };

struct TrainingItem { String path; int label; };
std::vector<TrainingItem> myTrainingData;

// ======================================================================
// SERIAL HELPER: blank Enter repeats the last command
// A \r\n pair counts as a single line end so a normal send does not double-fire.
// ======================================================================
struct SerialRepeatState {
  char lastCmd = 0;
  bool lineHadCommand = false;
  bool prevWasLineEnd = false;
  char resolve(char rawByte){
    if (rawByte=='\r' || rawByte=='\n') {
      char result = 0;
      if (!prevWasLineEnd) {
        if (!lineHadCommand && lastCmd != 0) result = lastCmd;
        lineHadCommand = false;
      }
      prevWasLineEnd = true;
      return result;
    }
    prevWasLineEnd = false;
    lineHadCommand = true;
    lastCmd = rawByte;
    return rawByte;
  }
};

bool mySDavailable = false;
int  myMenuIndex = 1;
bool myIsSelected = false;
int  myLastSelectedClass = 0;      // class last visited in the menu; Train+Infer SNN labels reps with it
bool gVerboseInfer = true;         // debug trail on (true) or transcript only (false)

// Classes with no samples are never predicted.
bool gActiveClassMask[NUM_CLASSES];
bool gClassEverTrained[NUM_CLASSES] = {};   // trained live this power cycle (Train+Infer writes no CSV)

// Forward declarations (Arduino IDE needs these declared before first use)
void myDrawMenu();
void myResetMenuState();
void myCommitClass(int classIdx, bool mayRepeat);
void myResetOutputDebounceAnn();
void myUpdateOutputStringAnn(int pred, float prob);
void myResetGestureState();
void myFindStillClass();
int myFindClassByKey(char c);
void myValidateClassKeys();
bool myHandleFreezeCommand(char c);

// ======================================================================
// TOUCH INPUT
// A tap is counted on release and only if contact lasted >= minPressMs, with
// a 180 ms settle time, so contact bounce cannot fake a 3-tap exit.
// Returns from myCheckTouchInput(): 0 nothing, 1 single tap, 2 triple tap.
// ======================================================================
const int myThresholdPress = 1100, myThresholdRelease = 900;
struct TouchState {
  bool isTouching = false; int tapCount = 0;
  unsigned long pressStartTime=0, firstTapTime=0, lastReleaseTime=0, lastCheckTime=0;
  const unsigned long tapWindow=900;
  const unsigned long debounceDelay=180;
  const unsigned long minPressMs=15;
  const int longPressTaps=3;
};
TouchState myTouch;
unsigned long myLastActivityTime=0, myLastTapTime=0;
const int myTapCooldown = 250;

int myReadTouch() { int s=0; for(int i=0;i<3;i++){s+=analogRead(A0);delayMicroseconds(100);} return s/3; }
void myResetTouchState(){ myTouch.isTouching=false; myTouch.tapCount=0; myTouch.pressStartTime=0; myTouch.firstTapTime=0; myTouch.lastReleaseTime=0; myTouch.lastCheckTime=0; }
void myUpdateTouchState(){
  unsigned long now=millis();
  if (now - myTouch.lastCheckTime < 20) return;
  myTouch.lastCheckTime = now;
  int val = myReadTouch();
  bool active = myTouch.isTouching ? (val>myThresholdRelease) : (val>myThresholdPress);
  if (active && !myTouch.isTouching) {
    if (now - myTouch.lastReleaseTime < myTouch.debounceDelay) return;
    myTouch.isTouching = true;
    myTouch.pressStartTime = now;
  }
  if (!active && myTouch.isTouching) {
    myTouch.isTouching = false;
    myTouch.lastReleaseTime = now;
    if (now - myTouch.pressStartTime < myTouch.minPressMs) return;
    if (myTouch.tapCount==0 || (now-myTouch.firstTapTime<myTouch.tapWindow)) { if(myTouch.tapCount==0) myTouch.firstTapTime=now; myTouch.tapCount++; }
    else { myTouch.tapCount=1; myTouch.firstTapTime=now; }
  }
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

// ======================================================================
// UTILITIES
// ======================================================================
inline float myClip(float v,float mn=-100,float mx=100){ if(isnan(v)||isinf(v)) return 0; return constrain(v,mn,mx); }
inline float myLeakyRelu(float x){ return x>0?x:0.1f*x; }
inline float myLeakyReluDeriv(float x){ return x>0?1.0f:0.1f; }

// Uppercases a-z, leaves everything else (digits, symbols, already-upper) untouched.
// Used so class hotkeys match a letter regardless of case, while a symbol key like
// '!' or '@' still only matches itself.
inline char myUpperIfAlpha(char c){ return (c>='a'&&c<='z') ? c-32 : c; }

void mySoftmax(float* x,int size){
  float mx=x[0]; for(int i=1;i<size;i++) if(x[i]>mx) mx=x[i];
  float sum=0; for(int i=0;i<size;i++){x[i]=exp(x[i]-mx); sum+=x[i];}
  for(int i=0;i<size;i++) x[i]/=sum;
}
void myShuffle(std::vector<TrainingItem>& v){
  for(int i=(int)v.size()-1;i>0;i--){ int j=random(0,i+1); std::swap(v[i],v[j]); }
}
void myScaleArray(float* a,int n,float s){ for(int i=0;i<n;i++) a[i]*=s; }

// ANN input: (accel - mean) / std, clipped. The SNN does not use this.
void myNormalizeInput(float* buf){
  for(int t=0;t<IMU_TIMESTEPS;t++) for(int a=0;a<IMU_AXES;a++){
    int idx=t*IMU_AXES+a;
    buf[idx]=(buf[idx]-myAccelMean[a])/(myAccelStd[a]+1e-8f);
    buf[idx]=myClip(buf[idx],-5.0f,5.0f);
  }
}

// Calibration: hold the board still; mean/std are saved to SD and reused on later boots.
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

// ======================================================================
// SAFE WEIGHT-FILE SAVING
// Weights are written to "<file>.tmp" first and only renamed over the real
// file once the write is complete, so a power cut mid-save leaves the
// previous good weights untouched. On boot, a complete orphaned .tmp
// (power cut between remove and rename) is recovered automatically.
// ======================================================================
bool myFinishSave(const char* finalPath, size_t written, size_t expected){
  String tmp = String(finalPath) + ".tmp";
  if (written != expected) { SD.remove(tmp.c_str()); return false; }
  if (SD.exists(finalPath)) SD.remove(finalPath);
  return SD.rename(tmp.c_str(), finalPath);
}
void myRecoverTmp(const char* finalPath, size_t expectedBytes){
  if (!mySDavailable || SD.exists(finalPath)) return;
  String tmp = String(finalPath) + ".tmp";
  if (!SD.exists(tmp.c_str())) return;
  File f = SD.open(tmp.c_str(), FILE_READ); if (!f) return;
  size_t sz = f.size(); f.close();
  if (sz == expectedBytes) SD.rename(tmp.c_str(), finalPath);
}

// ======================================================================
// DATA ON THE SD CARD  (/motion/<class label>/s<N>.csv, 40 rows of ax,ay,az in g)
// ======================================================================
int myCountSamples(int classIdx){
  if(!mySDavailable) return 0;
  String path="/motion/"+String(myClasses[classIdx].folder);
  File root=SD.open(path); if(!root) return 0;
  int count=0; while(File f=root.openNextFile()){ if(!f.isDirectory()&&String(f.name()).endsWith(".csv")) count++; f.close(); }
  root.close(); return count;
}

void myRefreshActiveClassMask(){
  bool anyActive=false;
  for(int c=0;c<NUM_CLASSES;c++){ gActiveClassMask[c] = (myCountSamples(c) > 0) || gClassEverTrained[c]; if(gActiveClassMask[c]) anyActive=true; }
  if(!anyActive) for(int c=0;c<NUM_CLASSES;c++) gActiveClassMask[c]=true;
}
int myArgmaxActive(float* scores, int size){
  int best=-1;
  for(int j=0;j<size;j++){ if(!gActiveClassMask[j]) continue; if(best==-1 || scores[j]>scores[best]) best=j; }
  return (best==-1) ? 0 : best;
}

bool myCaptureSample(int classIdx){
  String folderPath="/motion/"+String(myClasses[classIdx].folder);
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
    String path="/motion/"+String(myClasses[c].folder);
    File root=SD.open(path); if(!root) continue;
    while(File file=root.openNextFile()){
      String name=file.name();
      if(!file.isDirectory()&&name.endsWith(".csv")){ myTrainingData.push_back({path+"/"+name,c}); classCounts[c]++; }
      file.close();
    }
    root.close();
  }
}

// Shared by the training screens: split off VALIDATION_SAMPLES per class for testing.
int mySplitValidation(std::vector<TrainingItem>& valData){
  int valCount=0;
  myShuffle(myTrainingData);
  if(VALIDATION_SAMPLES>0){
    int held[NUM_CLASSES]={}; std::vector<TrainingItem> trainOnly;
    for(auto& it: myTrainingData){ if(held[it.label]<VALIDATION_SAMPLES){valData.push_back(it);held[it.label]++;valCount++;} else trainOnly.push_back(it); }
    myTrainingData=trainOnly;
  }
  return valCount;
}

// Read one gesture (40 x 3 raw g values) from a CSV file.
bool myReadCsvRep(const char* path,float* outBuf){
  File f=SD.open(path); if(!f) return false;
  for(int t=0;t<IMU_TIMESTEPS;t++){
    for(int a=0;a<IMU_AXES;a++){ outBuf[t*IMU_AXES+a]=f.parseFloat(); if(a<IMU_AXES-1) while(f.available()&&f.peek()==',') f.read(); }
    while(f.available()&&(f.peek()=='\n'||f.peek()=='\r')) f.read();
  }
  f.close(); return true;
}

// ======================================================================
// ANN MODEL: windowed Conv1D -> Dense -> Dense -> Output, backprop + Adam
// ======================================================================
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
float* myAnnRawBuf=nullptr;

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
  if(!myReadCsvRep(path,outBuf)) return false;
  myNormalizeInput(outBuf); return true;
}

#define ANN_WEIGHT_FILE "/header/myAnnWeights.bin"
const size_t ANN_FILE_BYTES = (size_t)(CONV1_WEIGHTS_ANN+CONV1_FILTERS+DENSE1_WEIGHTS_ANN+DENSE1_SIZE+DENSE2_WEIGHTS+DENSE2_SIZE+OUTPUT_WEIGHTS+NUM_CLASSES)*4;
bool myAnnLoadWeights(){
  if(!mySDavailable) return false;
  myRecoverTmp(ANN_WEIGHT_FILE, ANN_FILE_BYTES);
  if(!SD.exists(ANN_WEIGHT_FILE)) return false;
  File f=SD.open(ANN_WEIGHT_FILE,FILE_READ); if(!f) return false;
  if(f.size()!=ANN_FILE_BYTES){ f.close(); return false; }
  f.read((uint8_t*)gAnn.conv1_w,CONV1_WEIGHTS_ANN*4); f.read((uint8_t*)gAnn.conv1_b,CONV1_FILTERS*4);
  f.read((uint8_t*)gAnn.dense1_w,DENSE1_WEIGHTS_ANN*4); f.read((uint8_t*)gAnn.dense1_b,DENSE1_SIZE*4);
  f.read((uint8_t*)gAnn.dense2_w,DENSE2_WEIGHTS*4); f.read((uint8_t*)gAnn.dense2_b,DENSE2_SIZE*4);
  f.read((uint8_t*)gAnn.output_w,OUTPUT_WEIGHTS*4); f.read((uint8_t*)gAnn.output_b,NUM_CLASSES*4);
  f.close(); gAnn.trained=true; return true;
}
// Returns true only if the complete file was written and renamed into place.
bool myAnnSaveWeights(){
  if(!mySDavailable) return false;
  if(!SD.exists("/header")) SD.mkdir("/header");
  String tmp = String(ANN_WEIGHT_FILE) + ".tmp";
  File f=SD.open(tmp.c_str(),FILE_WRITE);
  if(!f) return false;
  size_t w=0;
  w+=f.write((uint8_t*)gAnn.conv1_w,CONV1_WEIGHTS_ANN*4); w+=f.write((uint8_t*)gAnn.conv1_b,CONV1_FILTERS*4);
  w+=f.write((uint8_t*)gAnn.dense1_w,DENSE1_WEIGHTS_ANN*4); w+=f.write((uint8_t*)gAnn.dense1_b,DENSE1_SIZE*4);
  w+=f.write((uint8_t*)gAnn.dense2_w,DENSE2_WEIGHTS*4); w+=f.write((uint8_t*)gAnn.dense2_b,DENSE2_SIZE*4);
  w+=f.write((uint8_t*)gAnn.output_w,OUTPUT_WEIGHTS*4); w+=f.write((uint8_t*)gAnn.output_b,NUM_CLASSES*4);
  f.close();
  return myFinishSave(ANN_WEIGHT_FILE, w, ANN_FILE_BYTES);
}

// ======================================================================
// SNN MODEL: streaming, causal, surrogate-gradient trained
// ======================================================================
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

// Per-tick recording of one gesture, needed for BPTT.
float*   rConv1MemPre;   float* rPooled;
uint8_t* rConv1Spike;
float*   rDense1MemPre;  uint8_t* rDense1Spike;
float*   rDense2MemPre;  uint8_t* rDense2Spike;
float*   rOutCurrent;    float* rTrace;
float*   mySnnSpikeInput;   // the encoded gesture: IMU_TIMESTEPS x SNN_CHANNELS
float*   mySnnRawBuf;       // scratch: one raw gesture in g

// Live streaming state (used by training and continuous inference).
float   gStreamRing[CONV1_KERNEL][SNN_CHANNELS];
int     gStreamRingFill = 0;
int     gStreamTickParity = 0;
float   gStreamConv1MemPost[CONV1_FILTERS];
float   gStreamPooledHeld[CONV1_FILTERS];
float   gStreamDense1MemPost[DENSE1_SIZE];
float   gStreamDense2MemPost[DENSE2_SIZE];
float   gStreamTrace[NUM_CLASSES];
uint8_t gConvSpikePrevTick[CONV1_FILTERS];
long    gStreamConv1Spikes=0, gStreamDense1Spikes=0, gStreamDense2Spikes=0, gStreamTicks=0;   // diagnostics

// ---- Spike encoder: accelerometer (g) -> ON/OFF spikes per axis + idle spike ----
// An axis fires when it has moved more than SPIKE_DELTA_G from where it last
// fired, then its reference moves to the new value. Fast movement -> spikes
// nearly every tick, slow movement -> occasional spikes, sensor noise -> none.
// Works directly in g so it does not depend on calibration.
struct SpikeEncoder {
  float ref[IMU_AXES]; bool primed=false;
  void reset(){ primed=false; }
  void step(const float* g, float* out){
    if(!primed){ for(int a=0;a<IMU_AXES;a++) ref[a]=g[a]; primed=true; }
    bool any=false;
    for(int a=0;a<IMU_AXES;a++){
      float d=g[a]-ref[a];
      bool on = d> SPIKE_DELTA_G, off = d< -SPIKE_DELTA_G;
      out[a*2+0]=on?1.0f:0.0f; out[a*2+1]=off?1.0f:0.0f;
      if(on||off){ ref[a]=g[a]; any=true; }
    }
    out[SNN_IDLE_CH]=any?0.0f:1.0f;   // "nothing moved" is information too
  }
};

inline float mySurrogateDeriv(float memPre){
  float d = 1.0f + SURROGATE_K * fabsf(memPre - LIF_THRESHOLD);
  return 1.0f / (d * d);
}

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
  mySnnRawBuf=(float*)ps_malloc(IMU_TIMESTEPS*IMU_AXES*sizeof(float));

  // He initialisation (spiking layers need enough initial current to fire)
  float c1=sqrt(2.0f/(CONV1_KERNEL*SNN_CHANNELS));
  for(int i=0;i<CONV1_WEIGHTS_SNN;i++) gSnn.conv1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*c1;
  float d1=sqrt(2.0f/CONV1_FILTERS);
  for(int i=0;i<DENSE1_WEIGHTS_SNN;i++) gSnn.dense1_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d1;
  float d2=sqrt(2.0f/DENSE1_SIZE);
  for(int i=0;i<DENSE2_WEIGHTS;i++) gSnn.dense2_w[i]=((float)rand()/RAND_MAX-0.5f)*2*d2;
  float od=sqrt(2.0f/DENSE2_SIZE);
  for(int i=0;i<OUTPUT_WEIGHTS;i++) gSnn.output_w[i]=((float)rand()/RAND_MAX-0.5f)*2*od;
}

// Restart the network as if a fresh gesture were beginning (spike counters are kept).
void mySnnResetDynamics(){
  memset(gStreamRing,0,sizeof(gStreamRing));
  gStreamRingFill=0; gStreamTickParity=0;
  memset(gStreamConv1MemPost,0,sizeof(gStreamConv1MemPost));
  memset(gStreamPooledHeld,0,sizeof(gStreamPooledHeld));
  memset(gStreamDense1MemPost,0,sizeof(gStreamDense1MemPost));
  memset(gStreamDense2MemPost,0,sizeof(gStreamDense2MemPost));
  memset(gStreamTrace,0,sizeof(gStreamTrace));
}
void mySnnResetStreamState(){
  mySnnResetDynamics();
  gStreamConv1Spikes=gStreamDense1Spikes=gStreamDense2Spikes=gStreamTicks=0;
}

// One tick of the network. spikeSample = SNN_CHANNELS values (0/1) for this instant.
// If record==true the internal state is stored at index tIdx for BPTT.
// Returns the current best class (argmax of the trace, inactive classes excluded).
int mySnnTick(float* spikeSample, bool record, int tIdx){
  for(int k=0;k<CONV1_KERNEL-1;k++) memcpy(gStreamRing[k],gStreamRing[k+1],SNN_CHANNELS*sizeof(float));
  memcpy(gStreamRing[CONV1_KERNEL-1],spikeSample,SNN_CHANNELS*sizeof(float));
  if (gStreamRingFill < CONV1_KERNEL) gStreamRingFill++;
  gStreamTicks++;

  // conv1: last 5 ticks (oldest..newest, zero-padded until full) -> LIF
  uint8_t curConvSpike[CONV1_FILTERS];
  int missing = CONV1_KERNEL - gStreamRingFill;
  for(int f=0;f<CONV1_FILTERS;f++){
    float current = gSnn.conv1_b[f];
    for(int k=missing;k<CONV1_KERNEL;k++)
      for(int a=0;a<SNN_CHANNELS;a++)
        current += gStreamRing[k][a] * gSnn.conv1_w[(k*SNN_CHANNELS+a)*CONV1_FILTERS+f];
    float memPre = gStreamConv1MemPost[f]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rConv1MemPre[tIdx*CONV1_FILTERS+f]=memPre; rConv1Spike[tIdx*CONV1_FILTERS+f]=spike; }
    if (spike) gStreamConv1Spikes++;
    gStreamConv1MemPost[f] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    curConvSpike[f] = spike;
  }

  // pooling: OR each pair of conv ticks; the result is held for 2 ticks
  if (gStreamTickParity == 0) {
    memcpy(gConvSpikePrevTick, curConvSpike, sizeof(curConvSpike));
  } else {
    for (int f=0; f<CONV1_FILTERS; f++)
      gStreamPooledHeld[f] = (gConvSpikePrevTick[f] | curConvSpike[f]) ? 1.0f : 0.0f;
  }
  gStreamTickParity = 1 - gStreamTickParity;
  if (record) for (int f=0; f<CONV1_FILTERS; f++) rPooled[tIdx*CONV1_FILTERS+f] = gStreamPooledHeld[f];

  // dense1 (LIF)
  uint8_t d1spike[DENSE1_SIZE];
  for(int j=0;j<DENSE1_SIZE;j++){
    float current = gSnn.dense1_b[j];
    for(int i=0;i<CONV1_FILTERS;i++) current += gStreamPooledHeld[i] * gSnn.dense1_w[i*DENSE1_SIZE+j];
    float memPre = gStreamDense1MemPost[j]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rDense1MemPre[tIdx*DENSE1_SIZE+j]=memPre; rDense1Spike[tIdx*DENSE1_SIZE+j]=spike; }
    if (spike) gStreamDense1Spikes++;
    gStreamDense1MemPost[j] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    d1spike[j]=spike;
  }

  // dense2 (LIF)
  uint8_t d2spike[DENSE2_SIZE];
  for(int j=0;j<DENSE2_SIZE;j++){
    float current = gSnn.dense2_b[j];
    for(int i=0;i<DENSE1_SIZE;i++) current += d1spike[i] * gSnn.dense2_w[i*DENSE2_SIZE+j];
    float memPre = gStreamDense2MemPost[j]*LIF_LEAK + current;
    uint8_t spike = (memPre >= LIF_THRESHOLD) ? 1 : 0;
    if (record){ rDense2MemPre[tIdx*DENSE2_SIZE+j]=memPre; rDense2Spike[tIdx*DENSE2_SIZE+j]=spike; }
    if (spike) gStreamDense2Spikes++;
    gStreamDense2MemPost[j] = memPre - (spike ? LIF_THRESHOLD : 0.0f);
    d2spike[j]=spike;
  }

  // output: plain current (no spiking) feeding a leaky per-class trace
  for(int c=0;c<NUM_CLASSES;c++){
    float current = gSnn.output_b[c];
    for(int i=0;i<DENSE2_SIZE;i++) current += d2spike[i] * gSnn.output_w[i*NUM_CLASSES+c];
    gStreamTrace[c] = gStreamTrace[c]*SNN_TRACE_LEAK + current;
    if (record){ rOutCurrent[tIdx*NUM_CLASSES+c]=current; rTrace[tIdx*NUM_CLASSES+c]=gStreamTrace[c]; }
  }
  return myArgmaxActive(gStreamTrace,NUM_CLASSES);
}

// ---- Surrogate-gradient backprop-through-time ---------------------------
// Runs after a recorded 40-tick pass. Softmax cross-entropy is measured on
// every tick from TRAIN_LOSS_FROM_TICK onward (so the trace learns to stay on
// the right class). The spike's derivative is replaced by a smooth curve
// (mySurrogateDeriv); the reset subtraction is treated as constant.
// Gradients are ADDED to gSnn.*_grad. Zero them before a batch, apply Adam after.
void mySnnZeroGrad(){
  memset(gSnn.conv1_w_grad,0,CONV1_WEIGHTS_SNN*sizeof(float));   memset(gSnn.conv1_b_grad,0,CONV1_FILTERS*sizeof(float));
  memset(gSnn.dense1_w_grad,0,DENSE1_WEIGHTS_SNN*sizeof(float)); memset(gSnn.dense1_b_grad,0,DENSE1_SIZE*sizeof(float));
  memset(gSnn.dense2_w_grad,0,DENSE2_WEIGHTS*sizeof(float));     memset(gSnn.dense2_b_grad,0,DENSE2_SIZE*sizeof(float));
  memset(gSnn.output_w_grad,0,OUTPUT_WEIGHTS*sizeof(float));     memset(gSnn.output_b_grad,0,NUM_CLASSES*sizeof(float));
}
void mySnnScaleGrad(float s){
  myScaleArray(gSnn.conv1_w_grad,CONV1_WEIGHTS_SNN,s);   myScaleArray(gSnn.conv1_b_grad,CONV1_FILTERS,s);
  myScaleArray(gSnn.dense1_w_grad,DENSE1_WEIGHTS_SNN,s); myScaleArray(gSnn.dense1_b_grad,DENSE1_SIZE,s);
  myScaleArray(gSnn.dense2_w_grad,DENSE2_WEIGHTS,s);     myScaleArray(gSnn.dense2_b_grad,DENSE2_SIZE,s);
  myScaleArray(gSnn.output_w_grad,OUTPUT_WEIGHTS,s);     myScaleArray(gSnn.output_b_grad,NUM_CLASSES,s);
}

void mySnnBackwardBPTT(int label, float* lossOut){
  float dTrace[NUM_CLASSES]; memset(dTrace,0,sizeof(dTrace));
  const float lossW = 1.0f / (IMU_TIMESTEPS - TRAIN_LOSS_FROM_TICK);
  float lossTotal = 0;

  float dDense2MemPost[DENSE2_SIZE]; memset(dDense2MemPost,0,sizeof(dDense2MemPost));
  float dDense1MemPost[DENSE1_SIZE]; memset(dDense1MemPost,0,sizeof(dDense1MemPost));
  float dConv1MemPost[CONV1_FILTERS]; memset(dConv1MemPost,0,sizeof(dConv1MemPost));
  static float dPooledAccum[IMU_TIMESTEPS][CONV1_FILTERS];
  memset(dPooledAccum,0,sizeof(dPooledAccum));

  // output trace -> dense2 -> dense1, newest tick first
  for (int t = IMU_TIMESTEPS-1; t >= 0; t--) {
    if (t >= TRAIN_LOSS_FROM_TICK) {
      float p[NUM_CLASSES];
      for(int c=0;c<NUM_CLASSES;c++) p[c]=rTrace[t*NUM_CLASSES+c];
      mySoftmax(p,NUM_CLASSES);
      lossTotal += -log(max(p[label],1e-7f)) * lossW;
      for(int c=0;c<NUM_CLASSES;c++) dTrace[c] += (p[c]-(c==label?1.0f:0.0f)) * lossW;
    }
    float dOutCurrent[NUM_CLASSES];
    for(int c=0;c<NUM_CLASSES;c++) dOutCurrent[c]=dTrace[c];
    for(int c=0;c<NUM_CLASSES;c++) dTrace[c]*=SNN_TRACE_LEAK;

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
      dDense2MemPre[j] = dDense2Spike[j]*sg + dDense2MemPost[j];
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

  // Send each pooled-feature gradient back to the two conv ticks that were OR'ed
  // (OR is not differentiable, so both get the full gradient, like max-pool routing).
  static float dConvSpikeFromPool[IMU_TIMESTEPS][CONV1_FILTERS];
  memset(dConvSpikeFromPool,0,sizeof(dConvSpikeFromPool));
  for(int t=1;t<IMU_TIMESTEPS;t++){
    int pairStart = 2*((t-1)/2);
    for(int f=0;f<CONV1_FILTERS;f++){
      dConvSpikeFromPool[pairStart][f]   += dPooledAccum[t][f];
      dConvSpikeFromPool[pairStart+1][f] += dPooledAccum[t][f];
    }
  }

  // conv1 backward through time
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
          if (srcT<0) continue;
          for(int a=0;a<SNN_CHANNELS;a++)
            gSnn.conv1_w_grad[(k*SNN_CHANNELS+a)*CONV1_FILTERS+f]+=mySnnSpikeInput[srcT*SNN_CHANNELS+a]*dConv1MemPre[f];
        }
      }
    }
    for(int f=0;f<CONV1_FILTERS;f++) dConv1MemPost[f]=dConv1MemPre[f]*LIF_LEAK;
  }
  if (lossOut) *lossOut = lossTotal;
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

// Weights file has its own name and a size check so an older, differently-shaped file is ignored.
// (Name kept from v015 - the network shape is unchanged, so existing weights still load.)
#define SNN_WEIGHT_FILE "/header/mySnnWeightsV15.bin"
const size_t SNN_FILE_BYTES = (size_t)(CONV1_WEIGHTS_SNN+CONV1_FILTERS+DENSE1_WEIGHTS_SNN+DENSE1_SIZE+DENSE2_WEIGHTS+DENSE2_SIZE+OUTPUT_WEIGHTS+NUM_CLASSES)*4;
bool mySnnLoadWeights(){
  if(!mySDavailable) return false;
  myRecoverTmp(SNN_WEIGHT_FILE, SNN_FILE_BYTES);
  if(!SD.exists(SNN_WEIGHT_FILE)) return false;
  File f=SD.open(SNN_WEIGHT_FILE,FILE_READ); if(!f) return false;
  if(f.size()!=SNN_FILE_BYTES){ f.close(); return false; }
  f.read((uint8_t*)gSnn.conv1_w,CONV1_WEIGHTS_SNN*4);   f.read((uint8_t*)gSnn.conv1_b,CONV1_FILTERS*4);
  f.read((uint8_t*)gSnn.dense1_w,DENSE1_WEIGHTS_SNN*4); f.read((uint8_t*)gSnn.dense1_b,DENSE1_SIZE*4);
  f.read((uint8_t*)gSnn.dense2_w,DENSE2_WEIGHTS*4);     f.read((uint8_t*)gSnn.dense2_b,DENSE2_SIZE*4);
  f.read((uint8_t*)gSnn.output_w,OUTPUT_WEIGHTS*4);     f.read((uint8_t*)gSnn.output_b,NUM_CLASSES*4);
  f.close(); gSnn.trained=true; return true;
}
// Returns true only if the complete file was written and renamed into place.
bool mySnnSaveWeights(){
  if(!mySDavailable) return false;
  if(!SD.exists("/header")) SD.mkdir("/header");
  String tmp = String(SNN_WEIGHT_FILE) + ".tmp";
  File f=SD.open(tmp.c_str(),FILE_WRITE);
  if(!f) return false;
  size_t w=0;
  w+=f.write((uint8_t*)gSnn.conv1_w,CONV1_WEIGHTS_SNN*4);   w+=f.write((uint8_t*)gSnn.conv1_b,CONV1_FILTERS*4);
  w+=f.write((uint8_t*)gSnn.dense1_w,DENSE1_WEIGHTS_SNN*4); w+=f.write((uint8_t*)gSnn.dense1_b,DENSE1_SIZE*4);
  w+=f.write((uint8_t*)gSnn.dense2_w,DENSE2_WEIGHTS*4);     w+=f.write((uint8_t*)gSnn.dense2_b,DENSE2_SIZE*4);
  w+=f.write((uint8_t*)gSnn.output_w,OUTPUT_WEIGHTS*4);     w+=f.write((uint8_t*)gSnn.output_b,NUM_CLASSES*4);
  f.close();
  return myFinishSave(SNN_WEIGHT_FILE, w, SNN_FILE_BYTES);
}

// Training-time variation: shift the gesture in time and change its size slightly,
// so the network learns the shape rather than one exact timing.
void myAugmentRep(float* buf){
  float tmp[IMU_TIMESTEPS*IMU_AXES], avg[IMU_AXES]={0,0,0};
  int   shift = random(-AUG_MAX_SHIFT, AUG_MAX_SHIFT+1);
  float gain  = 1.0f + ((float)random(-100,101)/100.0f) * AUG_GAIN_RANGE;
  for(int t=0;t<IMU_TIMESTEPS;t++) for(int a=0;a<IMU_AXES;a++) avg[a]+=buf[t*IMU_AXES+a]/IMU_TIMESTEPS;
  for(int t=0;t<IMU_TIMESTEPS;t++){
    int src = constrain(t+shift,0,IMU_TIMESTEPS-1);
    for(int a=0;a<IMU_AXES;a++) tmp[t*IMU_AXES+a] = avg[a] + (buf[src*IMU_AXES+a]-avg[a])*gain;
  }
  memcpy(buf,tmp,sizeof(tmp));
}

// Load a CSV gesture and turn it into spikes in mySnnSpikeInput.
bool mySnnLoadSampleEncoded(const char* path, bool augment){
  if(!myReadCsvRep(path,mySnnRawBuf)) return false;
  if(augment) myAugmentRep(mySnnRawBuf);
  SpikeEncoder enc; enc.reset();
  for(int t=0;t<IMU_TIMESTEPS;t++) enc.step(&mySnnRawBuf[t*IMU_AXES], &mySnnSpikeInput[t*SNN_CHANNELS]);
  return true;
}

// ======================================================================
// AIR-WRITING OUTPUT ENGINE
// Turns confirmed class predictions into a running line of text.
//   "Still" -> one space (never a leading or doubled space)
//   "Delete" -> remove last character; may repeat every DELETE_REPEAT_MS
//   other  -> append the class letter. The same letter may follow itself in both
//             modes. SNN: each letter comes from one complete gesture (see
//             GESTURE SEGMENTATION up top), so one gesture = one letter.
//             ANN: every confident 1-second window types, repeats included.
// Shared by both inference modes, so switching mid-line cannot double-type.
// ======================================================================
String myOutputString = "";
int gLastCommittedClass = -1;
unsigned long gLastCommitMs = 0;

String myClassText(int idx){ return String(myClasses[idx].label); }

// Word-wrapped tail of the transcript on the 72x40 OLED.
void myDrawOutputOLED(){
  const int charsPerLine = 13, maxLines = 5;
  String s = myOutputString;
  int total = s.length();
  int start = max(0, total - charsPerLine*maxLines);
  u8g2.firstPage();
  do{
    u8g2.setFont(u8g2_font_5x7_tf);
    if (total == 0) { u8g2.drawStr(0,7,"(empty)"); }
    else {
      int y=7;
      for(int i=start; i<total && y<=39; i+=charsPerLine){
        String line = s.substring(i, min(i+charsPerLine, total));
        u8g2.drawStr(0,y,line.c_str());
        y+=8;
      }
    }
  } while(u8g2.nextPage());
}
// Always prints (quiet mode still shows the transcript).
void myOnOutputChanged(){ Serial.println(myOutputString); myDrawOutputOLED(); }

// Types/deletes for a class the engine is confident about. mayRepeat: true
// lets a letter follow itself (both SNN and ANN pass true). Still never
// repeats; Delete repeats after DELETE_REPEAT_MS. After any commit the SNN
// dynamics are reset, as if a fresh gesture were about to begin.
void myCommitClass(int classIdx, bool mayRepeat){
  String text = myClassText(classIdx);
  bool isDelete = (text == "Delete");
  bool isStill  = (text == "Still");
  if (classIdx == gLastCommittedClass) {
    bool deleteAgain = isDelete && (millis()-gLastCommitMs >= DELETE_REPEAT_MS);
    bool letterAgain = !isDelete && !isStill && mayRepeat;
    if (!deleteAgain && !letterAgain) return;
  }
  gLastCommittedClass = classIdx;
  gLastCommitMs = millis();

  if (text == "Still") {
    if (myOutputString.length() > 0 && myOutputString.charAt(myOutputString.length()-1) != ' ') {
      myOutputString += ' ';
      myOnOutputChanged();
    }
  } else if (isDelete) {
    if (myOutputString.length() > 0) {
      myOutputString.remove(myOutputString.length()-1);
      myOnOutputChanged();
    }
  } else {
    myOutputString += text;
    myOnOutputChanged();
  }
  mySnnResetDynamics();
}

// ---- SNN path: gesture segmentation ----------------------------------
// See the GESTURE SEGMENTATION notes near the top of the file. Motion is
// read straight from the spike encoder's own idle channel: a tick is
// "moving" whenever any axis spiked (the idle channel did NOT fire).
enum GestureState { GESTURE_IDLE, GESTURE_ACTIVE, GESTURE_ENDING, GESTURE_CLASSIFY, GESTURE_COOLDOWN };
GestureState gGestureState = GESTURE_IDLE;

int gGestureTicks     = 0;   // ticks elapsed since the current gesture's onset
int gOnsetMoveStreak  = 0;   // consecutive moving ticks seen while IDLE (onset detector)
int gStillStreak      = 0;   // consecutive still ticks seen while ACTIVE/ENDING (end detector)
int gIdleStillStreak  = 0;   // consecutive still ticks seen while IDLE (drives the auto-space)
int gSettleTicksLeft  = 0;   // RESET_SETTLE_TICKS countdown during COOLDOWN

// Whole-gesture evidence, reset at onset and read once at gesture end.
float gGestureProbSum[NUM_CLASSES];                    // running sum of per-tick probabilities
float gGestureProbMax[NUM_CLASSES];                    // highest single-tick probability seen
int   gGestureEvidenceTicks = 0;                        // ticks counted into the sums above
float gFinalRing[FINAL_PORTION_TICKS][NUM_CLASSES];    // last few ticks' probabilities (ring buffer)
int   gFinalRingPos = 0, gFinalRingFill = 0;
int   gLastPrintedBest = -1;                            // for throttling the verbose ACTIVE trail
float gGestureBuffer[MAX_GESTURE_TICKS][SNN_CHANNELS];  // raw spikes for the current gesture
                                                         // (not used for classification yet - kept for
                                                         // diagnostics and as a foundation for future work,
                                                         // e.g. exporting real gesture-length training data)

int gLiveBestClass = 0; float gLiveBestProb = 0;  // rolling "current guess" during ACTIVE - never typed
int gStillClassIdx = -1;                          // index of the "Still" class (space); -1 if none defined

// Looks up the "Still" class once at boot, by label rather than by position,
// so the idle-timeout space still works even if myClasses[] is reordered.
void myFindStillClass(){
  gStillClassIdx = -1;
  for(int c=0;c<NUM_CLASSES;c++) if(String(myClasses[c].label)=="Still"){ gStillClassIdx=c; break; }
}

// Turns the live trace into a probability distribution, masking out any
// class with no training data yet (same masking myUpdateOutputStringAnn's
// SNN predecessor used - see NUM_CLASSES/gActiveClassMask above).
void mySnnMaskedProbs(float* outP){
  for(int c=0;c<NUM_CLASSES;c++) outP[c] = gActiveClassMask[c] ? gStreamTrace[c] : -1e30f;
  mySoftmax(outP, NUM_CLASSES);
}

void myResetGestureEvidence(){
  memset(gGestureProbSum,0,sizeof(gGestureProbSum));
  memset(gGestureProbMax,0,sizeof(gGestureProbMax));
  gGestureEvidenceTicks=0;
  gFinalRingPos=0; gFinalRingFill=0;
  gLastPrintedBest=-1;
}

// Called once per tick while a gesture is ACTIVE/ENDING. Keeps a running
// average for the whole gesture and a short ring of the most recent ticks
// (the "final portion") - see myClassifyGesture() for how they are blended.
void myGestureAccumulate(float* p, float* spikeTick){
  for(int c=0;c<NUM_CLASSES;c++){
    gGestureProbSum[c]+=p[c];
    if(p[c]>gGestureProbMax[c]) gGestureProbMax[c]=p[c];
  }
  gGestureEvidenceTicks++;
  memcpy(gFinalRing[gFinalRingPos], p, sizeof(float)*NUM_CLASSES);
  gFinalRingPos=(gFinalRingPos+1)%FINAL_PORTION_TICKS;
  if(gFinalRingFill<FINAL_PORTION_TICKS) gFinalRingFill++;
  int bufIdx = gGestureTicks-1;
  if (bufIdx>=0 && bufIdx<MAX_GESTURE_TICKS) memcpy(gGestureBuffer[bufIdx], spikeTick, sizeof(float)*SNN_CHANNELS);
}

// The ONE decision for a completed gesture. Blends the whole-gesture average
// probability with the average over just the final portion, then requires
// both a minimum duration and clear confidence before typing anything.
// "Still" is deliberately excluded here - it has no motion to segment, so it
// can only ever be produced by the idle-stillness timeout in myGestureTick().
void myClassifyGesture(){
  float avgP[NUM_CLASSES], finalP[NUM_CLASSES], blended[NUM_CLASSES];
  int n = max(gGestureEvidenceTicks,1);
  for(int c=0;c<NUM_CLASSES;c++) avgP[c]=gGestureProbSum[c]/n;
  int fn = max(gFinalRingFill,1);
  for(int c=0;c<NUM_CLASSES;c++){
    float s=0; for(int r=0;r<gFinalRingFill;r++) s+=gFinalRing[r][c];
    finalP[c]=s/fn;
  }
  for(int c=0;c<NUM_CLASSES;c++) blended[c]=(1.0f-GESTURE_FINAL_WEIGHT)*avgP[c]+GESTURE_FINAL_WEIGHT*finalP[c];

  int best=-1, second=-1;
  for(int c=0;c<NUM_CLASSES;c++){
    if(!gActiveClassMask[c] || c==gStillClassIdx) continue;
    if(best==-1 || blended[c]>blended[best]){ second=best; best=c; }
    else if(second==-1 || blended[c]>blended[second]) second=c;
  }
  float margin = (best>=0 && second>=0) ? blended[best]-blended[second] : (best>=0 ? blended[best] : 0.0f);

  if (gVerboseInfer) {
    Serial.print("[FINAL] ");
    for(int c=0;c<NUM_CLASSES;c++) if(gActiveClassMask[c] && c!=gStillClassIdx) Serial.printf("%s=%.2f ", myClasses[c].label, blended[c]);
    Serial.printf("margin=%.2f\n", margin);
  }

  bool longEnough = (gGestureTicks >= MIN_GESTURE_TICKS);
  bool confident  = (best>=0) && (blended[best] >= GESTURE_COMMIT_MIN_PROB) && (margin >= GESTURE_COMMIT_MARGIN);

  if (best>=0 && longEnough && confident) {
    if (gVerboseInfer) Serial.printf("[COMMIT] %s\n", myClasses[best].label);
    myCommitClass(best, true);   // internally resets the SNN dynamics
  } else {
    if (gVerboseInfer) Serial.printf("[COMMIT] none (%s)\n", !longEnough ? "gesture too short" : "not confident enough");
    mySnnResetDynamics();        // start the next gesture from a clean state either way
  }
}

// Call once per tick with this tick's raw spike sample and the tick counter
// (used only for the [GESTURE] START diagnostic line). This is the state
// machine described in GESTURE SEGMENTATION near the top of the file.
void myGestureTick(float* spikeTick, unsigned long currentTick){
  bool moving = (spikeTick[SNN_IDLE_CH] < 0.5f);

  switch(gGestureState){

    case GESTURE_COOLDOWN:
      if(--gSettleTicksLeft <= 0) gGestureState = GESTURE_IDLE;
      break;

    case GESTURE_IDLE: {
      if(moving){ gOnsetMoveStreak++; gIdleStillStreak=0; }
      else       { gOnsetMoveStreak=0; gIdleStillStreak++; }

      if(gOnsetMoveStreak >= GESTURE_ONSET_TICKS){
        gGestureState = GESTURE_ACTIVE;
        gGestureTicks = gOnsetMoveStreak;   // the onset ticks already counted as part of the gesture
        gStillStreak = 0;
        myResetGestureEvidence();
        if (gVerboseInfer) Serial.printf("[GESTURE] START tick=%lu\n", currentTick);
      } else if (gStillClassIdx>=0 && gIdleStillStreak >= STILL_HOLD_TICKS) {
        // Sustained stillness with nothing else going on = a space. Safe to
        // call every tick past the threshold: myCommitClass() is a no-op if
        // "Still" is already the last thing that was typed.
        myCommitClass(gStillClassIdx, true);
      }
      break;
    }

    case GESTURE_ACTIVE:
    case GESTURE_ENDING: {
      gGestureTicks++;
      float p[NUM_CLASSES]; mySnnMaskedProbs(p);
      myGestureAccumulate(p, spikeTick);
      gLiveBestClass = myArgmaxActive(p, NUM_CLASSES);
      gLiveBestProb  = p[gLiveBestClass];

      if(moving){ gStillStreak=0; gGestureState=GESTURE_ACTIVE; }
      else       { gStillStreak++; gGestureState=GESTURE_ENDING; }

      // Diagnostics only - this never types anything (see item 8/myCommitClass).
      if (gVerboseInfer && (gLiveBestClass != gLastPrintedBest || gGestureTicks % 4 == 0)) {
        Serial.printf("[GESTURE] ACTIVE tick=%d best=%s p=%.2f\n", gGestureTicks, myClasses[gLiveBestClass].label, gLiveBestProb);
        gLastPrintedBest = gLiveBestClass;
      }

      bool endedByStillness = (gStillStreak >= GESTURE_END_STILL_TICKS);
      bool endedByMaxLength = (gGestureTicks >= MAX_GESTURE_TICKS);
      if (endedByStillness || endedByMaxLength) {
        if (gVerboseInfer) Serial.printf("[GESTURE] END duration=%d stillTicks=%d%s\n", gGestureTicks, gStillStreak, endedByMaxLength?" (max length reached)":"");
        myClassifyGesture();
        gGestureState = GESTURE_COOLDOWN;
        gSettleTicksLeft = RESET_SETTLE_TICKS;
        gOnsetMoveStreak = 0;
      }
      break;
    }
  }
}

// Called on entering "Infer SNN". Does NOT touch gLastCommittedClass - that
// is shared with the ANN path so switching modes mid-line cannot double-type.
void myResetGestureState(){
  gGestureState = GESTURE_IDLE;
  gGestureTicks=0; gOnsetMoveStreak=0; gStillStreak=0; gIdleStillStreak=0; gSettleTicksLeft=0;
  myResetGestureEvidence();
  gLiveBestClass=0; gLiveBestProb=0;
}

// ---- ANN path: one decision per 1-second window ----
// A window whose top probability is below ANN_COMMIT_MIN_PROB types nothing.
// Otherwise its class is typed once ANN_CONFIRM_WINDOWS windows in a row agree
// (default 1 = every confident window). After a typed letter the count restarts,
// so the same letter can be typed again by the next window(s).
int gAnnPendingClass = -1, gAnnPendingWindows = 0;
void myResetOutputDebounceAnn(){ gAnnPendingClass=-1; gAnnPendingWindows=0; }
void myUpdateOutputStringAnn(int pred, float prob){
  if (prob < ANN_COMMIT_MIN_PROB) { gAnnPendingClass=-1; gAnnPendingWindows=0; return; }
  if (pred == gAnnPendingClass) gAnnPendingWindows++;
  else { gAnnPendingClass = pred; gAnnPendingWindows = 1; }
  if (gAnnPendingWindows < ANN_CONFIRM_WINDOWS) return;
  myCommitClass(pred, true);       // true: the same letter may follow itself
  gAnnPendingWindows = 0;          // next letter needs its own confirmation
}

// ======================================================================
// ACTIONS: DATA COLLECTION
// ======================================================================
// One capture: 1 s warning, record 1 s, report. progressSuffix e.g. " (auto 3/10)".
bool myCaptureAndReport(int classIdx, int &cc, const char* progressSuffix){
  Serial.print("[Collect] Recording in 1s... ");
  delay(1000);
  Serial.println("Now!");
  if(myCaptureSample(classIdx)){
    cc++;
    Serial.printf("[Collect] Saved #%d for '%s'%s.\n", cc, myClasses[classIdx].label, progressSuffix);
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,myClasses[classIdx].label);char b[20];snprintf(b,20,"Saved: %d",cc);u8g2.drawStr(0,20,b);u8g2.drawStr(0,32,"TAP=More");}while(u8g2.nextPage());
    return true;
  } else {
    Serial.println("[Collect] Capture FAILED (SD write error).");
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"SD write");u8g2.drawStr(0,18,"FAILED");}while(u8g2.nextPage());
    delay(800);
    return false;
  }
}

#define AUTO_COLLECT_COUNT 10   // gestures per auto-collect burst
// The collect-screen key list (Serial) + OLED. Shown on entry.
void myDrawCollectScreen(int classIdx, int cc){
  Serial.println("Keys in this collect screen:");
  Serial.println("  t  (or 1 TAP)  = capture ONE 1-second gesture");
  Serial.printf ("  a              = AUTO-COLLECT: capture %d gestures in a row (press once, then keep making the gesture)\n", AUTO_COLLECT_COUNT);
  Serial.println("  Enter (blank)  = repeat your last command");
  Serial.println("  l  (or 3x TAP) = leave and go back to the menu");
  u8g2.firstPage();
  do{
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0,7,myClasses[classIdx].label);
    u8g2.drawStr(0,15,"TAP=Capture");
    u8g2.drawStr(0,23,"a=Auto x10");
    u8g2.drawStr(0,31,"3xTap=Exit");
    char b[20]; snprintf(b,20,"Count: %d",cc); u8g2.drawStr(0,39,b);
  }while(u8g2.nextPage());
}

// 3x tap or 'l' stops the burst early. Bytes typed during the burst are fed
// through serialRepeat so a trailing Enter does not upset the blank-Enter repeat.
void myActionAutoCollect(int classIdx, int &cc, SerialRepeatState& serialRepeat){
  Serial.println();
  Serial.println("*** 'a' PRESSED = AUTO-COLLECT MODE ***");
  Serial.printf ("[Collect] Will capture %d gestures in a row for '%s' (about 2 s each: 1 s get-ready + 1 s recording).\n", AUTO_COLLECT_COUNT, myClasses[classIdx].label);
  Serial.println("[Collect] Stop early with 'l' or 3x TAP. Starting in 1s...");
  u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"AUTO-COLLECT");u8g2.drawStr(0,18,myClasses[classIdx].label);u8g2.drawStr(0,28,"x10 starting");}while(u8g2.nextPage());
  delay(1000);
  int done=0;
  for(int i=0;i<AUTO_COLLECT_COUNT;i++){
    if(myCheckTouchInput()==2){ Serial.printf("[Collect] Auto-collect stopped early (3x tap) after %d/%d.\n", done, AUTO_COLLECT_COUNT); return; }
    while(Serial.available()){ char c=serialRepeat.resolve(Serial.read()); if(c=='l'||c=='L'){ Serial.printf("[Collect] Auto-collect stopped early ('l') after %d/%d.\n", done, AUTO_COLLECT_COUNT); return; } }
    char suffix[16]; snprintf(suffix,16," (auto %d/%d)", i+1, AUTO_COLLECT_COUNT);
    { char b[20]; snprintf(b,20,"AUTO %d/%d",i+1,AUTO_COLLECT_COUNT);
      u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,b);u8g2.drawStr(0,18,myClasses[classIdx].label);u8g2.drawStr(0,28,"Get ready...");}while(u8g2.nextPage()); }
    if(!myCaptureAndReport(classIdx, cc, suffix)){
      Serial.printf("[Collect] Auto-collect aborted after %d/%d due to a write failure.\n", done, AUTO_COLLECT_COUNT);
      return;
    }
    done++;
  }
  Serial.printf("[Collect] Auto-collect complete: %d new samples saved for '%s' (total now %d).\n", done, myClasses[classIdx].label, cc);
  u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"AUTO done");char b[20];snprintf(b,20,"Count: %d",cc);u8g2.drawStr(0,20,b);u8g2.drawStr(0,32,"a=10 more");}while(u8g2.nextPage());
}

void myActionCollect(int classIdx){
  if(!mySDavailable){ Serial.println("[Collect] No SD card - can't collect samples."); myResetMenuState(); return; }
  myResetTouchState();
  int cc=myCountSamples(classIdx);
  Serial.printf("\n--- Collecting class '%s' (existing samples: %d) ---\n", myClasses[classIdx].label, cc);
  myDrawCollectScreen(classIdx, cc);
  SerialRepeatState serialRepeat;
  while(true){
    bool doCapture=false;
    if(Serial.available()){
      char c=serialRepeat.resolve(Serial.read());
      if (c) {
        if(c=='l'||c=='L'){ Serial.printf("[Collect] Exiting '%s' with %d samples saved.\n", myClasses[classIdx].label, cc); myResetMenuState(); return; }
        if(c=='t'||c=='T') doCapture=true;
        if(c=='a'||c=='A') myActionAutoCollect(classIdx, cc, serialRepeat);
      }
    }
    int ta=myCheckTouchInput();
    if(ta==2){ Serial.printf("[Collect] Exiting '%s' with %d samples saved.\n", myClasses[classIdx].label, cc); myResetMenuState(); return; }
    if(ta==1) doCapture=true;
    if(doCapture) myCaptureAndReport(classIdx, cc, "");
  }
}

// ======================================================================
// TRAINING HELPERS
// ======================================================================
// Used only by the continuous (auto) trainers: true if the user asked to stop
// ('l' on Serial, or 3x tap on the touch pad).
bool myTrainAbortRequested(){
  if(myCheckTouchInput()==2) return true;
  while(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L') return true; }
  return false;
}

// ======================================================================
// ACTIONS: ANN TRAIN / INFER
// ======================================================================
// One full training run of TARGET_EPOCHS epochs, then save. Returns false if the
// user aborted part-way (nothing is saved in that case). In continuous mode the
// validation split is passed in so it stays the same from loop to loop.
bool myTrainAnnOnce(std::vector<TrainingItem>& valData, int valCount, bool continuous, const char* tag,
                    float& bestAccOut, int& bestEpochOut, bool& savedOut){
  unsigned long trainStartMs = millis();
  float bestAcc=-1; int bestEpoch=-1;
  savedOut=false;
  const int nTrain = max((int)myTrainingData.size(),1);
  for(int epoch=0;epoch<TARGET_EPOCHS;epoch++){
    myShuffle(myTrainingData);
    float loss=0; int correct=0, processed=0; myAnnZeroGrad();
    for(int si=0;si<(int)myTrainingData.size();si++){
      if(continuous){ if(myTrainAbortRequested()) return false; } else myCheckTouchBackground();
      if(!myAnnLoadSample(myTrainingData[si].path.c_str(),myAnnRawBuf)) continue;
      myAnnForward(myAnnRawBuf);
      int label=myTrainingData[si].label;
      loss+=-log(max(myAnnFinal[label],1e-7f));
      int pred=0; for(int j=1;j<NUM_CLASSES;j++) if(myAnnFinal[j]>myAnnFinal[pred]) pred=j;
      if(pred==label) correct++;
      myAnnBackward(myAnnRawBuf,label);
      gClassEverTrained[label]=true;
      processed++;
      if((si+1)%BATCH_SIZE==0 || si==(int)myTrainingData.size()-1){
        float sc=1.0f/processed;
        myScaleArray(gAnn.conv1_w_grad,CONV1_WEIGHTS_ANN,sc);   myScaleArray(gAnn.conv1_b_grad,CONV1_FILTERS,sc);
        myScaleArray(gAnn.dense1_w_grad,DENSE1_WEIGHTS_ANN,sc); myScaleArray(gAnn.dense1_b_grad,DENSE1_SIZE,sc);
        myScaleArray(gAnn.dense2_w_grad,DENSE2_WEIGHTS,sc);     myScaleArray(gAnn.dense2_b_grad,DENSE2_SIZE,sc);
        myScaleArray(gAnn.output_w_grad,OUTPUT_WEIGHTS,sc);     myScaleArray(gAnn.output_b_grad,NUM_CLASSES,sc);
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
    float trainAcc = 100.0f*correct/nTrain;
    float valAcc=0;
    if(valCount>0){ int vc=0; for(auto& vi: valData){ if(!myAnnLoadSample(vi.path.c_str(),myAnnRawBuf)) continue; myAnnForward(myAnnRawBuf); int p=0; for(int j=1;j<NUM_CLASSES;j++) if(myAnnFinal[j]>myAnnFinal[p]) p=j; if(p==vi.label) vc++; } valAcc=100.0f*vc/valCount; }
    float score = (valCount>0) ? valAcc : trainAcc;
    bool isBest = score>bestAcc;
    if(isBest){ bestAcc=score; bestEpoch=epoch+1; }
    Serial.printf("[%s] Epoch %d/%d Loss=%.4f TrainAcc=%.1f%% ValAcc=%.1f%%%s\n",tag,epoch+1,TARGET_EPOCHS,loss/nTrain,trainAcc,valAcc, isBest?"  <- best so far":"");
    u8g2.firstPage();
    do{ u8g2.setFont(u8g2_font_5x7_tf); char b[24];
        snprintf(b,24,"%s Ep %d/%d",continuous?"ANN*":"ANN",epoch+1,TARGET_EPOCHS); u8g2.drawStr(0,8,b);
        snprintf(b,24,"Tr%.0f Val%.0f",trainAcc,valAcc); u8g2.drawStr(0,20,b);
    } while(u8g2.nextPage());
  }
  gAnn.trained=true;
  savedOut = myAnnSaveWeights();
  Serial.printf("[%s] Done in %.1fs. Best %s=%.1f%% at epoch %d. Weights %s.\n",
                tag,(millis()-trainStartMs)/1000.0f, valCount>0?"ValAcc":"TrainAcc", bestAcc, bestEpoch, savedOut?"saved":"NOT SAVED (SD write failed)");
  bestAccOut=bestAcc; bestEpochOut=bestEpoch;
  return true;
}

// continuous=false : one run of TARGET_EPOCHS, save, back to menu (menu key G).
// continuous=true  : repeat forever - TARGET_EPOCHS, save, again - until 'l' / 3x tap (menu key a).
void myActionTrainAnn(bool continuous){
  const char* tag = continuous ? "ANN-auto" : "ANN";
  if(!mySDavailable){ Serial.printf("[%s] No SD card - can't train.\n", tag); myResetMenuState(); return; }
  if(continuous) Serial.println("\n=== AUTO-TRAIN ANN (continuous: train, save, repeat) ===");
  else           Serial.println("\n=== Train ANN (windowed Conv1D+Dense, backprop+Adam) ===");
  int cc[NUM_CLASSES]={}; myBuildTrainingList(cc);
  myRefreshActiveClassMask();
  Serial.printf("[%s] Samples found: ", tag);
  for(int c=0;c<NUM_CLASSES;c++) Serial.printf("%s=%d%s ", myClasses[c].label, cc[c], cc[c]==0?"(empty)":"");
  Serial.printf("(total=%d)\n", (int)myTrainingData.size());
  if(myTrainingData.empty()){ Serial.printf("[%s] No training samples - collect data first. Aborting.\n", tag); u8g2.firstPage(); do{u8g2.drawStr(0,15,"No samples!");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  std::vector<TrainingItem> valData;
  int valCount = mySplitValidation(valData);
  Serial.printf("[%s] Training on %d gestures, holding out %d for validation. %d epochs per loop, lr=%.4f, batch=%d.\n",
                tag,(int)myTrainingData.size(), valCount, TARGET_EPOCHS, LEARNING_RATE, BATCH_SIZE);
  Serial.printf("[%s] Frozen layers: conv1=%d dense1=%d dense2=%d output=%d\n",
                tag,gAnn.freezeConv1, gAnn.freezeDense1, gAnn.freezeDense2, gAnn.freezeOutput);

  float bestAcc=0; int bestEpoch=0; bool saved=false;
  if(!continuous){
    myTrainAnnOnce(valData, valCount, false, tag, bestAcc, bestEpoch, saved);
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,10,"ANN trained");char b[20];snprintf(b,20,"Best:%.0f%% ep%d",bestAcc,bestEpoch);u8g2.drawStr(0,24,b);}while(u8g2.nextPage()); delay(1500); myResetMenuState();
    return;
  }

  Serial.println("[ANN-auto] Each loop = train all epochs, then save weights to the SD card, then start again.");
  Serial.println("[ANN-auto] Stop with 'l' or 3x TAP. The loop in progress is discarded; every finished loop is already saved.");
  myResetTouchState();
  int savedLoops=0, loopNo=0;
  while(true){
    loopNo++;
    Serial.printf("\n[ANN-auto] ---- Loop %d starting (%d epochs) ----\n", loopNo, TARGET_EPOCHS);
    if(!myTrainAnnOnce(valData, valCount, true, tag, bestAcc, bestEpoch, saved)){
      Serial.printf("[ANN-auto] Stopped. Loop %d was unfinished and discarded. %d loop(s) saved on the SD card this run.\n", loopNo, savedLoops);
      if(savedLoops>0){ myAnnLoadWeights(); Serial.println("[ANN-auto] Reloaded the last saved weights so memory matches the SD card."); }
      u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,10,"ANN auto stop");char b[20];snprintf(b,20,"%d loops saved",savedLoops);u8g2.drawStr(0,24,b);}while(u8g2.nextPage()); delay(1500);
      myResetMenuState(); return;
    }
    if(saved) savedLoops++;
    else Serial.println("[ANN-auto] WARNING: this loop could not be saved to the SD card.");
    Serial.printf("[ANN-auto] Loop %d complete (%d saved so far). Starting the next loop...\n", loopNo, savedLoops);
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"ANN auto");char b[20];snprintf(b,20,"Loop %d %s",loopNo,saved?"saved":"NOSAVE");u8g2.drawStr(0,20,b);snprintf(b,20,"Best:%.0f%%",bestAcc);u8g2.drawStr(0,32,b);}while(u8g2.nextPage());
  }
}

// Windowed inference: one prediction per 1-second window, typed if confident (no voting).
void myActionInferAnn(){
  if(!gAnn.trained){ Serial.println("[ANN] No trained weights yet - run 'Train ANN' first."); u8g2.firstPage(); do{u8g2.drawStr(0,15,"No ANN weights");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  Serial.println("\n--- Infer ANN (one decision per 1-second window, repeats allowed) ---");
  Serial.printf("[ANN] Verbose inference output: %s ('v' in the menu toggles it)\n", gVerboseInfer?"ON":"OFF");
  myRefreshActiveClassMask();
  myResetOutputDebounceAnn();
  Serial.printf("[ANN] Output so far: \"%s\"\n", myOutputString.c_str());
  Serial.println("Move the sensor. 3x TAP (or 'l') returns to the menu.");
  if (!gVerboseInfer) myDrawOutputOLED();
  int windowCount=0;
  while(true){
    if(myCheckTouchInput()==2){ Serial.printf("[ANN] Exiting after %d windows.\n", windowCount); myResetMenuState(); return; }
    if(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L'){ Serial.printf("[ANN] Exiting after %d windows.\n", windowCount); myResetMenuState();return;} }
    for(int t=0;t<IMU_TIMESTEPS;t++){
      unsigned long tS=millis();
      myAnnRawBuf[t*IMU_AXES+0]=myIMU.readFloatAccelX(); myAnnRawBuf[t*IMU_AXES+1]=myIMU.readFloatAccelY(); myAnnRawBuf[t*IMU_AXES+2]=myIMU.readFloatAccelZ();
      long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
    }
    myNormalizeInput(myAnnRawBuf); myAnnForward(myAnnRawBuf);
    int pred=myArgmaxActive(myAnnFinal,NUM_CLASSES);
    float prob=myAnnFinal[pred];
    windowCount++;
    myUpdateOutputStringAnn(pred, prob);
    if (gVerboseInfer) {
      Serial.printf("[ANN] #%d pred=%s p=%.2f%s\n",windowCount,myClasses[pred].label,prob,prob<ANN_COMMIT_MIN_PROB?" (too unsure, not typed)":"");
      u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"ANN:");u8g2.drawStr(0,18,myClasses[pred].label);char b[20];snprintf(b,20,"p=%.2f",prob);u8g2.drawStr(0,28,b);}while(u8g2.nextPage());
    }
  }
}

// ======================================================================
// ACTIONS: SNN TRAIN / INFER
// ======================================================================
// One full SNN training run of TARGET_EPOCHS epochs, then save. Returns false if the
// user aborted part-way (nothing is saved in that case). Gradients are averaged over
// BATCH_SIZE gestures per weight update. Training gestures get random time
// shift / size changes each epoch; validation gestures do not.
bool myTrainSnnOnce(std::vector<TrainingItem>& valData, int valCount, bool continuous, const char* tag,
                    float& bestAccOut, int& bestEpochOut, bool& savedOut){
  unsigned long trainStartMs = millis();
  float bestAcc=-1; int bestEpoch=-1;
  savedOut=false;
  const int nTrain = max((int)myTrainingData.size(),1);
  for(int epoch=0; epoch<TARGET_EPOCHS; epoch++){
    myShuffle(myTrainingData);
    float lossSum=0; int correct=0, inBatch=0;
    mySnnZeroGrad();
    for(size_t si=0; si<myTrainingData.size(); si++){
      if(continuous){ if(myTrainAbortRequested()) return false; } else myCheckTouchBackground();
      if(!mySnnLoadSampleEncoded(myTrainingData[si].path.c_str(), true)) continue;
      mySnnResetStreamState();
      int label = myTrainingData[si].label;
      int pred=0;
      for(int t=0;t<IMU_TIMESTEPS;t++) pred = mySnnTick(&mySnnSpikeInput[t*SNN_CHANNELS], true, t);
      if (pred==label) correct++;

      float sampleLoss=0;
      mySnnBackwardBPTT(label, &sampleLoss);
      lossSum += sampleLoss;
      gClassEverTrained[label]=true;
      inBatch++;
      if (inBatch>=BATCH_SIZE || si==myTrainingData.size()-1) {
        mySnnScaleGrad(1.0f/inBatch);
        mySnnApplyAdam();
        mySnnZeroGrad(); inBatch=0;
      }
    }
    float trainAcc = 100.0f*correct/nTrain;
    float valAcc=0;
    if(valCount>0){
      int vc=0;
      for(auto& vi: valData){
        if(!mySnnLoadSampleEncoded(vi.path.c_str(), false)) continue;
        mySnnResetStreamState();
        int pred=0;
        for(int t=0;t<IMU_TIMESTEPS;t++) pred=mySnnTick(&mySnnSpikeInput[t*SNN_CHANNELS], false, 0);
        if (pred==vi.label) vc++;
      }
      valAcc=100.0f*vc/valCount;
    }
    float score = (valCount>0) ? valAcc : trainAcc;
    bool isBest = score>bestAcc;
    if(isBest){ bestAcc=score; bestEpoch=epoch+1; }
    const char* spikeHint = "";
    if (gStreamConv1Spikes==0 && gStreamDense1Spikes==0) spikeHint = "  [WARN spikes dead - lower LIF_THRESHOLD]";
    else if (gStreamConv1Spikes > (long)(IMU_TIMESTEPS*CONV1_FILTERS*0.9f)) spikeHint = "  [WARN spiking every tick - raise LIF_THRESHOLD]";
    Serial.printf("[%s] Epoch %d/%d Loss=%.4f TrainAcc=%.1f%% ValAcc=%.1f%%%s (last-gesture spikes: c1=%ld d1=%ld d2=%ld)%s\n",
      tag,epoch+1,TARGET_EPOCHS, lossSum/nTrain, trainAcc, valAcc, isBest?"  <- best so far":"",
      gStreamConv1Spikes, gStreamDense1Spikes, gStreamDense2Spikes, spikeHint);
    u8g2.firstPage();
    do{ u8g2.setFont(u8g2_font_5x7_tf); char b[24];
        snprintf(b,24,"%s Ep %d/%d",continuous?"SNN*":"SNN",epoch+1,TARGET_EPOCHS); u8g2.drawStr(0,8,b);
        snprintf(b,24,"Tr%.0f Val%.0f",trainAcc,valAcc); u8g2.drawStr(0,20,b);
    } while(u8g2.nextPage());
  }
  gSnn.trained=true;
  savedOut = mySnnSaveWeights();
  Serial.printf("[%s] Done in %.1fs. Best %s=%.1f%% at epoch %d. Weights %s.\n",
                tag,(millis()-trainStartMs)/1000.0f, valCount>0?"ValAcc":"TrainAcc", bestAcc, bestEpoch, savedOut?"saved":"NOT SAVED (SD write failed)");
  bestAccOut=bestAcc; bestEpochOut=bestEpoch;
  return true;
}

// continuous=false : one run of TARGET_EPOCHS, save, back to menu (menu key J).
// continuous=true  : repeat forever - TARGET_EPOCHS, save, again - until 'l' / 3x tap (menu key A).
void myActionTrainSnn(bool continuous){
  const char* tag = continuous ? "SNN-auto" : "SNN";
  if(!mySDavailable){ Serial.printf("[%s] No SD card - can't train.\n", tag); myResetMenuState(); return; }
  if(continuous) Serial.println("\n=== AUTO-TRAIN SNN (continuous: train, save, repeat) ===");
  else           Serial.println("\n=== Train SNN (streaming, causal, surrogate-gradient BPTT) ===");
  int cc[NUM_CLASSES]={}; myBuildTrainingList(cc);
  myRefreshActiveClassMask();
  Serial.printf("[%s] Samples found: ", tag);
  for(int c=0;c<NUM_CLASSES;c++) Serial.printf("%s=%d%s ", myClasses[c].label, cc[c], cc[c]==0?"(empty)":"");
  Serial.printf("(total=%d)\n", (int)myTrainingData.size());
  if(myTrainingData.empty()){ Serial.printf("[%s] No training samples - collect data first. Aborting.\n", tag); u8g2.firstPage(); do{u8g2.drawStr(0,15,"No samples!");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  std::vector<TrainingItem> valData;
  int valCount = mySplitValidation(valData);
  Serial.printf("[%s] Training on %d gestures, holding out %d for validation. %d epochs per loop, lr=%.4f, batch=%d.\n",
                tag,(int)myTrainingData.size(), valCount, TARGET_EPOCHS, LEARNING_RATE, BATCH_SIZE);
  Serial.printf("[%s] Frozen layers: conv1=%d dense1=%d dense2=%d output=%d | LIF_THRESHOLD=%.2f LIF_LEAK=%.2f SPIKE_DELTA_G=%.3f\n",
                tag,gSnn.freezeConv1, gSnn.freezeDense1, gSnn.freezeDense2, gSnn.freezeOutput, LIF_THRESHOLD, LIF_LEAK, SPIKE_DELTA_G);

  float bestAcc=0; int bestEpoch=0; bool saved=false;
  if(!continuous){
    myTrainSnnOnce(valData, valCount, false, tag, bestAcc, bestEpoch, saved);
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,10,"SNN trained");char b[20];snprintf(b,20,"Best:%.0f%% ep%d",bestAcc,bestEpoch);u8g2.drawStr(0,24,b);}while(u8g2.nextPage()); delay(1500); myResetMenuState();
    return;
  }

  Serial.println("[SNN-auto] Each loop = train all epochs, then save weights to the SD card, then start again.");
  Serial.println("[SNN-auto] Stop with 'l' or 3x TAP. The loop in progress is discarded; every finished loop is already saved.");
  myResetTouchState();
  int savedLoops=0, loopNo=0;
  while(true){
    loopNo++;
    Serial.printf("\n[SNN-auto] ---- Loop %d starting (%d epochs) ----\n", loopNo, TARGET_EPOCHS);
    if(!myTrainSnnOnce(valData, valCount, true, tag, bestAcc, bestEpoch, saved)){
      Serial.printf("[SNN-auto] Stopped. Loop %d was unfinished and discarded. %d loop(s) saved on the SD card this run.\n", loopNo, savedLoops);
      if(savedLoops>0){ mySnnLoadWeights(); Serial.println("[SNN-auto] Reloaded the last saved weights so memory matches the SD card."); }
      u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,10,"SNN auto stop");char b[20];snprintf(b,20,"%d loops saved",savedLoops);u8g2.drawStr(0,24,b);}while(u8g2.nextPage()); delay(1500);
      myResetMenuState(); return;
    }
    if(saved) savedLoops++;
    else Serial.println("[SNN-auto] WARNING: this loop could not be saved to the SD card.");
    Serial.printf("[SNN-auto] Loop %d complete (%d saved so far). Starting the next loop...\n", loopNo, savedLoops);
    u8g2.firstPage(); do{u8g2.setFont(u8g2_font_5x7_tf);u8g2.drawStr(0,8,"SNN auto");char b[20];snprintf(b,20,"Loop %d %s",loopNo,saved?"saved":"NOSAVE");u8g2.drawStr(0,20,b);snprintf(b,20,"Best:%.0f%%",bestAcc);u8g2.drawStr(0,32,b);}while(u8g2.nextPage());
  }
}

// Infer SNN: continuous, one tick at a time. The SNN itself still streams every
// tick (see mySnnTick below), but myGestureTick() gates what actually gets typed -
// see GESTURE SEGMENTATION near the top of the file.
void myActionInferSnnContinuous(){
  if(!gSnn.trained){ Serial.println("[SNN] No trained weights yet - run 'Train SNN' or 'Train+Infer SNN' first."); u8g2.firstPage(); do{u8g2.drawStr(0,15,"No SNN weights");}while(u8g2.nextPage()); delay(1200); myResetMenuState(); return; }
  Serial.println("\n--- Infer SNN (gesture-segmented: draw a letter, pause, it commits) ---");
  Serial.printf("[SNN-live] Verbose inference output: %s ('v' in the menu toggles it)\n", gVerboseInfer?"ON":"OFF");
  myRefreshActiveClassMask();
  myResetGestureState();
  Serial.printf("[SNN-live] Output so far: \"%s\"\n", myOutputString.c_str());
  Serial.println("Move the sensor. 3x TAP (or 'l') returns to the menu.");
  if (!gVerboseInfer) myDrawOutputOLED();
  mySnnResetStreamState();
  SpikeEncoder enc; enc.reset();
  unsigned long tickCount=0;
  long prevC1=0, prevD1=0, prevD2=0;
  while(true){
    if(myCheckTouchInput()==2){ Serial.printf("[SNN-live] Exiting after %lu ticks.\n", tickCount); myResetMenuState(); return; }
    if(Serial.available()){ char c=Serial.read(); if(c=='l'||c=='L'){ Serial.printf("[SNN-live] Exiting after %lu ticks.\n", tickCount); myResetMenuState();return;} }

    unsigned long tS=millis();
    float raw[IMU_AXES]={myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ()};
    float spikeTick[SNN_CHANNELS];
    enc.step(raw, spikeTick);

    mySnnTick(spikeTick, false, 0);   // the SNN always streams, regardless of gesture state
    tickCount++;
    myGestureTick(spikeTick, tickCount);   // decides START/ACTIVE/END/COMMIT - see above

    if (tickCount % 8 == 0) {   // ~every 200 ms: a health check on the spiking layers, independent of gestures
      float c1Rate=(gStreamConv1Spikes-prevC1)/8.0f, d1Rate=(gStreamDense1Spikes-prevD1)/8.0f, d2Rate=(gStreamDense2Spikes-prevD2)/8.0f;
      prevC1=gStreamConv1Spikes; prevD1=gStreamDense1Spikes; prevD2=gStreamDense2Spikes;
      if (gVerboseInfer) {
        Serial.printf("[SNN-live] tick=%lu spikes/tick: c1=%.2f d1=%.2f d2=%.2f\n", tickCount, c1Rate, d1Rate, d2Rate);
        u8g2.firstPage();
        do{
          u8g2.setFont(u8g2_font_5x7_tf);
          const char* stateLabel = (gGestureState==GESTURE_IDLE) ? "idle" : (gGestureState==GESTURE_COOLDOWN) ? "cooldown" : "drawing...";
          u8g2.drawStr(0,8, stateLabel);
          if (gGestureState==GESTURE_ACTIVE || gGestureState==GESTURE_ENDING) {
            u8g2.drawStr(0,18, myClasses[gLiveBestClass].label);
            char b[20]; snprintf(b,20,"p=%.2f",gLiveBestProb); u8g2.drawStr(0,28,b);
          } else {
            u8g2.drawStr(0,18, "(watching)");
          }
          char b2[20]; snprintf(b2,20,"sp%.1f/%.1f/%.1f",c1Rate,d1Rate,d2Rate); u8g2.drawStr(0,38,b2);
        } while(u8g2.nextPage());
      }
    }
    long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
  }
}

// Train+Infer SNN: streams live predictions during a 1-second gesture (labelled as the
// class last visited in the menu), then does one BPTT weight update from that gesture.
void myActionTrainInferSnn(){
  Serial.println("\n--- Train+Infer SNN (live predict during capture, then one BPTT step) ---");
  Serial.printf("[SNN Train+Infer] Labeling gestures as '%s' (visit a class item first to change). 1 TAP or 't' = start, 3x TAP or 'l' = exit.\n",
                myClasses[myLastSelectedClass].label);
  Serial.println("[SNN Train+Infer] Blank Enter repeats your last command.");
  myResetTouchState();
  SerialRepeatState serialRepeat;
  while(true){
    int ta = myCheckTouchInput();
    if (ta==2) { myResetMenuState(); return; }
    bool startRep = (ta==1);
    if (!startRep && Serial.available()){
      char c=serialRepeat.resolve(Serial.read());
      if(c=='l'||c=='L'){ myResetMenuState(); return; }
      if(c=='t'||c=='T') startRep=true;
    }
    if (!startRep) { delay(20); continue; }

    mySnnResetStreamState();
    SpikeEncoder enc; enc.reset();
    int label = myLastSelectedClass;
    int lastPred=0;
    for(int t=0;t<IMU_TIMESTEPS;t++){
      unsigned long tS=millis();
      float raw[IMU_AXES]={myIMU.readFloatAccelX(),myIMU.readFloatAccelY(),myIMU.readFloatAccelZ()};
      enc.step(raw, &mySnnSpikeInput[t*SNN_CHANNELS]);
      lastPred = mySnnTick(&mySnnSpikeInput[t*SNN_CHANNELS], true, t);
      if (t%5==0) Serial.printf("  live tick %d -> %s\n", t, myClasses[lastPred].label);
      long el=millis()-tS; if(el<SAMPLE_INTERVAL_MS) delay(SAMPLE_INTERVAL_MS-el);
    }

    mySnnZeroGrad();
    mySnnBackwardBPTT(label, nullptr);
    mySnnApplyAdam();
    gClassEverTrained[label]=true;
    myRefreshActiveClassMask();
    gSnn.trained = true;
    mySnnSaveWeights();

    Serial.printf("[SNN Train+Infer] Done. Labeled=%s LivePredAtEnd=%s (%s) | spikes: c1=%ld d1=%ld d2=%ld over %d ticks\n",
                  myClasses[label].label, myClasses[lastPred].label,
                  lastPred==label?"OK":"MISS",
                  gStreamConv1Spikes, gStreamDense1Spikes, gStreamDense2Spikes, IMU_TIMESTEPS);
    u8g2.firstPage();
    do{ u8g2.setFont(u8g2_font_5x7_tf); u8g2.drawStr(0,8,"Trained on:"); u8g2.drawStr(0,18,myClasses[label].label);
        u8g2.drawStr(0,28, lastPred==label ? "Pred: OK" : "Pred: MISS"); } while(u8g2.nextPage());
    delay(800);
  }
}

// ======================================================================
// MENU, STATUS AND SERIAL COMMANDS
// ======================================================================
// Returns the index of the class whose key matches c (letters compared
// case-insensitively, symbols compared exactly), or -1 if none match.
int myFindClassByKey(char c){
  char cu = myUpperIfAlpha(c);
  for(int i=0;i<NUM_CLASSES;i++) if (cu == myUpperIfAlpha(myClasses[i].key)) return i;
  return -1;
}

// Runs once at boot. Adding a class is just a new row in myClasses[], but if
// its key collides with another class (case-insensitively for letters) only
// the first match would ever fire - this warns on Serial so it gets noticed.
void myValidateClassKeys(){
  for(int i=0;i<NUM_CLASSES;i++){
    for(int j=i+1;j<NUM_CLASSES;j++){
      if (myUpperIfAlpha(myClasses[i].key) == myUpperIfAlpha(myClasses[j].key)){
        Serial.printf("WARNING: classes '%s' and '%s' share the hotkey '%c' - fix myClasses[]!\n",
                      myClasses[i].label, myClasses[j].label, myUpperIfAlpha(myClasses[i].key));
      }
    }
  }
}

// Freeze commands (use before training to test whether freezing early layers
// reduces interference when adding new classes).
// v016: conv1 moved from A/a to C/c because A/a now start continuous training.
// v022: moved again, from letters to punctuation, because letters are now
// used for classes. Returns true if c matched a freeze command (handled).
bool myHandleFreezeCommand(char c){
  switch(c){
    case '%': gSnn.freezeConv1=!gSnn.freezeConv1;   Serial.printf("SNN conv1 frozen=%d\n",gSnn.freezeConv1); return true;
    case '^': gSnn.freezeDense1=!gSnn.freezeDense1; Serial.printf("SNN dense1 frozen=%d\n",gSnn.freezeDense1); return true;
    case '&': gSnn.freezeDense2=!gSnn.freezeDense2; Serial.printf("SNN dense2 frozen=%d\n",gSnn.freezeDense2); return true;
    case '*': gSnn.freezeOutput=!gSnn.freezeOutput; Serial.printf("SNN output frozen=%d\n",gSnn.freezeOutput); return true;
    case '(': gAnn.freezeConv1=!gAnn.freezeConv1;   Serial.printf("ANN conv1 frozen=%d\n",gAnn.freezeConv1); return true;
    case ')': gAnn.freezeDense1=!gAnn.freezeDense1; Serial.printf("ANN dense1 frozen=%d\n",gAnn.freezeDense1); return true;
    case '-': gAnn.freezeDense2=!gAnn.freezeDense2; Serial.printf("ANN dense2 frozen=%d\n",gAnn.freezeDense2); return true;
    case '=': gAnn.freezeOutput=!gAnn.freezeOutput; Serial.printf("ANN output frozen=%d\n",gAnn.freezeOutput); return true;
  }
  return false;
}

char myMenuHotkey(int idx){
  if (idx<=NUM_CLASSES) return myClasses[idx-1].key;
  return myActionKeys[idx-NUM_CLASSES-1];
}
String myMenuLabel(int idx){ if(idx<=NUM_CLASSES) return String(myClasses[idx-1].label); return String(myActionLabels[idx-NUM_CLASSES-1]); }

void myPrintStatus(){
  Serial.println("\n=== STATUS ===");
  myRefreshActiveClassMask();
  Serial.print("Samples per class: ");
  for(int c=0;c<NUM_CLASSES;c++){ int n=myCountSamples(c); Serial.printf("%s=%d%s ", myClasses[c].label, n, n==0?"(empty)":""); }
  Serial.println();
  Serial.print("Active for prediction: ");
  for(int c=0;c<NUM_CLASSES;c++) if(gActiveClassMask[c]) Serial.printf("%s ", myClasses[c].label);
  Serial.println();
  Serial.printf("ANN: trained=%d  freeze conv1=%d dense1=%d dense2=%d output=%d  adamStep=%d\n",
                gAnn.trained, gAnn.freezeConv1, gAnn.freezeDense1, gAnn.freezeDense2, gAnn.freezeOutput, gAnn.adamStep);
  Serial.printf("SNN: trained=%d  freeze conv1=%d dense1=%d dense2=%d output=%d  adamStep=%d\n",
                gSnn.trained, gSnn.freezeConv1, gSnn.freezeDense1, gSnn.freezeDense2, gSnn.freezeOutput, gSnn.adamStep);
  Serial.printf("Training: lr=%.4f batch=%d epochs/loop=%d valSamples=%d\n", LEARNING_RATE, BATCH_SIZE, TARGET_EPOCHS, VALIDATION_SAMPLES);
  Serial.printf("SNN: LIF_THRESHOLD=%.2f LIF_LEAK=%.2f SNN_TRACE_LEAK=%.2f SPIKE_DELTA_G=%.3f\n", LIF_THRESHOLD, LIF_LEAK, SNN_TRACE_LEAK, SPIKE_DELTA_G);
  Serial.printf("Output engine (SNN gestures): onset=%d ticks, end-still=%d ticks, min=%d ticks, max=%d ticks, finalPortion=%d ticks (weight=%.2f), minProb=%.2f, margin=%.2f, settle=%d ticks, stillHold=%d ticks\n",
                GESTURE_ONSET_TICKS, GESTURE_END_STILL_TICKS, MIN_GESTURE_TICKS, MAX_GESTURE_TICKS, FINAL_PORTION_TICKS, GESTURE_FINAL_WEIGHT,
                GESTURE_COMMIT_MIN_PROB, GESTURE_COMMIT_MARGIN, RESET_SETTLE_TICKS, STILL_HOLD_TICKS);
  Serial.printf("Output engine (ANN): min prob=%.2f confirm=%d windows | Delete repeat=%lums\n", ANN_COMMIT_MIN_PROB, ANN_CONFIRM_WINDOWS, DELETE_REPEAT_MS);
  Serial.printf("Transcript: \"%s\"\n", myOutputString.c_str());
  Serial.printf("SD card: %s | Free PSRAM: %d bytes | Uptime: %lus\n", mySDavailable?"present":"absent", ESP.getFreePsram(), millis()/1000);
  Serial.println("Hotkeys: letters (case-insensitive)=class, !=Still, @=Delete, 1-7=actions, #=verbose, $=redraw menu, ?=status, ,=next, .=select");
  Serial.println("Freeze: % ^ & * = SNN conv1/dense1/dense2/output   ( ) - = = ANN conv1/dense1/dense2/output");
  Serial.println("==============");
}

void myResetMenuState(){ myIsSelected=false; myResetTouchState(); myLastActivityTime=millis(); myDrawMenu(); }
void myDrawMenu(){
  Serial.println("\n=== MENU ===");
  for(int i=1;i<=myTotalItems;i++) Serial.printf("%s%c. %s\n",(i==myMenuIndex)?" > ":"   ",myMenuHotkey(i),myMenuLabel(i).c_str());
  Serial.println("Hotkeys: letters (case-insensitive)=class (collect data), !=Still, @=Delete   1-5=Train ANN/Infer ANN/Train SNN/Infer SNN/Train+Infer SNN");
  Serial.println("Auto:    6=AUTO-TRAIN ANN  7=AUTO-TRAIN SNN  (continuous: 40 epochs, save to SD, repeat; 'l' or 3x TAP stops)");
  Serial.println("Inside a class: t=capture one  a=AUTO-COLLECT 10 in a row  l=exit");
  Serial.println("Other: % ^ & * = SNN freeze conv1/dense1/dense2/output   ( ) - = = ANN freeze conv1/dense1/dense2/output   ?=status  #=verbose toggle  $=reprint menu  ,=next  .=select");
  u8g2.firstPage();
  do{
    u8g2.setFont(u8g2_font_5x7_tf);
    u8g2.drawStr(0,7,"Tap=Next Hold=Ok");
    int start=max(1,myMenuIndex-1);
    for(int i=0;i<4;i++){ int cur=start+i; if(cur>myTotalItems) break; int y=17+i*8; String line=(cur==myMenuIndex?"> ":"  ")+myMenuLabel(cur); u8g2.drawStr(0,y,line.c_str()); }
  } while(u8g2.nextPage());
}
void myExecuteMenuItem(int idx){
  Serial.printf("[Menu] Selected: %c. %s\n", myMenuHotkey(idx), myMenuLabel(idx).c_str());
  if(idx<=NUM_CLASSES){ myLastSelectedClass = idx-1; myActionCollect(idx-1); return; }
  switch(idx-NUM_CLASSES-1){
    case 0: myActionTrainAnn(false); break;
    case 1: myActionInferAnn(); break;
    case 2: myActionTrainSnn(false); break;
    case 3: myActionInferSnnContinuous(); break;
    case 4: myActionTrainInferSnn(); break;
    case 5: myActionTrainAnn(true); break;
    case 6: myActionTrainSnn(true); break;
  }
}
void myHandleMenuNavigation(){
  unsigned long now=millis();
  if(!myIsSelected && Serial.available()){
    char c=Serial.read();
    if (myHandleFreezeCommand(c)) return;
    if (c=='?'){ myPrintStatus(); return; }
    if (c=='#'){ gVerboseInfer=!gVerboseInfer; Serial.printf("Verbose inference output=%d (%s)\n", gVerboseInfer, gVerboseInfer?"debug trail ON":"quiet - transcript only"); return; }
    if (c=='$'){ myDrawMenu(); return; }
    int classIdx = myFindClassByKey(c);
    if (classIdx>=0){ myMenuIndex=classIdx+1; myIsSelected=true; myExecuteMenuItem(myMenuIndex); return; }
    for(int a=0;a<NUM_ACTIONS;a++){
      if (c==myActionKeys[a]){ myMenuIndex=NUM_CLASSES+a+1; myIsSelected=true; myExecuteMenuItem(myMenuIndex); return; }
    }
    if(c==','){ if(now-myLastTapTime>myTapCooldown){ myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1; myDrawMenu(); myLastTapTime=now; } }
    else if(c=='.'){ myIsSelected=true; myExecuteMenuItem(myMenuIndex); }
  }
  if(!myIsSelected){
    int ta=myCheckTouchInput();
    if(ta==1){ if(now-myLastTapTime>myTapCooldown){ myMenuIndex++; if(myMenuIndex>myTotalItems) myMenuIndex=1; myDrawMenu(); myLastTapTime=now; } }
    else if(ta==2){ myIsSelected=true; myExecuteMenuItem(myMenuIndex); }
  }
}

// ======================================================================
// SETUP / LOOP
// ======================================================================
void setup(){
  Serial.begin(115200);
  while(!Serial && millis()<3000);
  delay(1000);
  Serial.println("\n=== XIAO ESP32-S3 Motion ML - Gesture-segmented SNN air-writing (imu-snn-v024) ===");
  Serial.println("Hotkeys: letters (case-insensitive) = class, ! = Still, @ = Delete, 1-7 = Train ANN/Infer ANN/Train SNN/Infer SNN/Train+Infer SNN/AutoTrain ANN/AutoTrain SNN, $ = menu, ? = status, # = verbose toggle.");

  randomSeed(esp_random());
  srand(esp_random());

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
  Serial.printf("Free PSRAM after allocation: %d bytes\n", ESP.getFreePsram());

  myAnnLoadWeights();
  if(!mySnnLoadWeights()) Serial.println("No compatible SNN weights found - run Train SNN (older SNN weights are not compatible).");
  myRefreshActiveClassMask();
  myValidateClassKeys();
  myFindStillClass();
  if (gStillClassIdx<0) Serial.println("NOTE: no class labelled 'Still' was found - the idle-timeout space is disabled.");

  myResetMenuState();
  Serial.println("System ready.");
}

void loop(){
  myHandleMenuNavigation();
}
