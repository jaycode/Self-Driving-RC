#include "car.h"
#include <Arduino.h>

Car::Car(const Motor& inEngine, const SteeringWheel& inSteer, const RC& inRemoteController) : 
         engine_(inEngine), steer_(inSteer), rc_(inRemoteController) {
}

void Car::accelerate(int strength) {
  // Remove this when we got the accelerometer later.
  curSpeed_ = strength;
  if (engineReverse_) {
    // For when we mixed up the cables.
    strength = -strength;
  }
  if (strength > 0) {
    engine_.forward(strength);
  }
  else if (strength < 0) {
    engine_.backward(-strength);
  }
  else if (strength == 0) {
    engine_.brake();
  }
}

void Car::brake() {
  engine_.brake();
}

void Car::listen() {
  uint8_t newMode = rc_.readDigital(RC_AUX1);

  if (newMode != curDriveMode_) {
    curDriveMode_ = newMode;
  }
  // Reads current steering angle, needed in listenAile().
  curSteerFeed_ = steer_.readFeed();

  // Update to use accelerometer output later.
//  curSpeed_ = 
  
  if (curDriveMode_ != DRIVE_MODE_AUTO) {
//     We set steer speed as 0 here to allow for two joysticks controlling the steering wheel.
    steerSpeed_ = 0;
    listenThro(); // throttling left joystick
    listenElev(); // throttling right joystick
    listenAile(); // steering right joystick
    accelerate(throSpeed_+elevSpeed_);
    steer(steerSpeed_);
  }
  else if (curDriveMode_ == DRIVE_MODE_AUTO) {
    // Communication happens in one direction, from computer to microcontroller.
    listenComputer();
    autoSteer();
  }

}

void Car::listenThro() {
  /**
   * The car can only move forward with throttle (due to how the remote control
   * was designed).
   */
  int valueRaw = rc_.readValue(RC_THRO);
  throSpeed_ = (int)normalize((float)valueRaw, minAThro_,
                         maxA_, engineMin_, engineMax_);

  if (throSpeed_ == engineMin_) {
    throSpeed_ = 0;
  }
}

void Car::listenElev() {
  /*
   * Only runs when there is no throttle. Useful for running in reverse.
   */
  int elev = rc_.readValue(RC_ELEV);
  if (elev == 0) {
    // Means the remote controller is off.
    elevSpeed_ = 0;
  } 
  else {
    elevSpeed_ = normalizeBi(elev, minA_, maxA_, engineMin_, engineMax_);
  }
}

void Car::listenAile() {
  int aile = rc_.readValue(RC_AILE);
  if (aile == 0) {
    // Means the remote controller is off.
    steerSpeed_ = 0;
  }
  else {
    steerSpeed_ = normalizeBi(aile, minA_, maxA_, steerMin_, steerMax_);
  }
}

void Car::steer(const int steerSpeed) {
  int ss = steerSpeed;
  ss = log10(fabs(ss))*30;
  
  if (steerSpeed < 0) {
    ss = -ss;
  }
  bool calc1 = curSteerFeed_ > steerFeedMax_-steerSlack_;
  bool calc2 = curSteerFeed_ < steerFeedMin_+steerSlack_;
  if (steerReverse_) {
    ss = -ss;
  }

  // The code below looks more complicated than needed to avoid
  // the steering wheel turning more than it should.
  if (ss > 0) {
    if (calc1) {
      steer_.brake();
    }
    else {
      steer_.forward(ss);
    }
  }
  else if (ss < 0) {
    if (calc2) {
      steer_.brake();
    }
    else {
      steer_.backward(-ss);
    }
  }
  else {
    steer_.brake();
  }
}

void Car::steerTo(int steerPos) {
  if (steerPos > steerFeedMax_) {
    steerPos = steerFeedMax_;
  }
  else if (steerPos < steerFeedMin_) {
    steerPos = steerFeedMin_;
  }
  targetSteer_ = steerPos;
}

void Car::autoSteer() {
  int slack = 30;
//  Serial.print("feed: ");
//  Serial.print(curSteerFeed_);
//  Serial.print(" target: ");
//  Serial.println(targetSteer_);
  if (curSteerFeed_ > targetSteer_ - slack &&
      curSteerFeed_ < targetSteer_ + slack) {
    steer(0);
  }
  else {
//    Serial.println("adjust");
    // To avoid oscillation, decrease the speed when closer to position.
    float speed = steerMax_ * fabs(curSteerFeed_ - targetSteer_) / float(steerFeedMax_ - steerFeedMin_);
    if (speed < 150) {
      speed /= 3;
    }
//    Serial.println(speed);
    if (curSteerFeed_ > targetSteer_) {
      steer(-speed);
    }
    else {
      steer(speed);
    }
  }
  
}

void Car::setCurDriveMode(uint8_t value) {
  curDriveMode_ = value;
}

// mode, velocity, steer
// m1v+010s0010
String Car::getStatus() {
  // --- Too slow ---
//  char cvelocity[3];
//  sprintf(cvelocity, "%03d", int(curSpeed_+0.5));
//  
//  String velocity;
//  if (curSpeed_ >= 0.0) {
//    velocity = String("+"+String(cvelocity));
//  }
//  else {
//    velocity = String(cvelocity);
//  }
//
//  char csteer[4];
//  sprintf(csteer, "%04d", curSteerFeed_);
//  String steer = String(csteer);
//  
//  String str = String("m"+String(curDriveMode_)+"v"+velocity+"s"+steer);
  // --- The code above was commented because it was too slow ---
  
  String str = String("m"+String(curDriveMode_)+"v"+String(curSpeed_, 0)+"s"+String(curSteerFeed_));
  return str;
}

//void Car::listenComputer() {
//  while (Serial.available() == 0);
//  char cmd = Serial.read();
//  Serial.print(cmd);
//  if (cmd == 's') {
//    for (int i=0; i<5; i++) {
//      while (Serial.available() == 0);
//      char pos = Serial.read();
//      Serial.print(pos);
//    }
//    Serial.print('\n');
//  }
//}

void Car::listenComputer() {
  while (Serial.available() == 0);
  char hostCmd = Serial.read();
  if (hostCmd == HOST_REQUEST_UPDATE) {
    sendCommand(DEV_STATUS, getStatus());
  }
  else if (hostCmd == HOST_AUTO_STEER) {
    String pos; // 0000 to 1023
    for (int i=0; i<5; i++) {
      while (Serial.available() == 0);
      char c = Serial.read();
//      Serial.print(c);
      pos += c;
    }
    int v = pos.toInt();
    steerTo(v);
//    Serial.print('\n');

//    while (Serial.available() < 5);
//    
//    byte num = Serial.readBytesUntil(';', pos, 5);
//    if (num > 0) {
//      String s = pos;
//      int v = s.toInt();
//      sendCommand(DEV_DEBUG, pos);
//      steerTo(v);
////        steerTo(100);
//
//    }
  }
  else if (hostCmd == HOST_AUTO_THROTTLE) {
    char thro[4]; // -255 to +255
//    while (Serial.available() < 5);
    byte num = Serial.readBytesUntil(';', thro, 5);
    if (num > 0) {
      String s = thro;
      int v = s.toInt();
      accelerate(v);
    }
  }
//  else {
//    // If it does not understand the command the computer gave,
//    // return the command back. This is needed to give a "kick"
//    // on the first loop.
//    sendCommand(DEV_DEBUG, hostCmd);
//  }
}
