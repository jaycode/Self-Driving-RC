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

  // Communication happens in one direction, from computer to microcontroller.
  listenComputer();
  if (newMode != curDriveMode_) {
//    sendCommand(CMD_CHANGE_DRIVE_MODE, newMode);
    curDriveMode_ = newMode;
  }
  // Reads current steering angle, needed in listenAile().
  curSteerFeed_ = steer_.readFeed();

  // Update to use accelerometer output later.
//  curSpeed_ = 
  
  if (curDriveMode_ != DRIVE_MODE_AUTO) {
    // We set steer speed as 0 here to allow for two joysticks controlling the steering wheel.
    steerSpeed_ = 0;
    listenThro(); // throttling left joystick
    listenElev(); // throttling right joystick
    listenAile(); // steering right joystick
    accelerate(throSpeed_+elevSpeed_);
    steer(steerSpeed_);
  }
  else if (curDriveMode_ == DRIVE_MODE_AUTO) {
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

String Car::getStatus() {
  String str = String("v"+String(curSpeed_, 4)+";o"+String(curSteerFeed_)+";");
  return str;
}

void Car::listenComputer() {
  char ccmd = Serial.read();
  if (ccmd == CCMD_DRIVE_MODE) {
    sendCommand(CMD_CHANGE_DRIVE_MODE, curDriveMode_);
  }
  else if (ccmd == CCMD_REQUEST_STATUS) {
    sendCommand(CMD_STATUS, getStatus());
  }
  else if (ccmd == CCMD_AUTO_STEER && curDriveMode_ == DRIVE_MODE_AUTO) {
    char pos[4]; // 0 to 1023
    byte num = Serial.readBytesUntil(';', pos, 4);
    if (num > 0) {
      String s = pos;
      int v = s.toInt();
//      sendCommand(CMD_DEBUG, v);
      steerTo(v);
    }
  }
  else if (ccmd == CCMD_AUTO_THROTTLE && curDriveMode_ == DRIVE_MODE_AUTO) {
    char thro[4]; // -255 to 255
    byte num = Serial.readBytesUntil(';', thro, 4);
    if (num > 0) {
      String s = thro;
      int v = s.toInt();
      accelerate(v);
//      sendCommand(CMD_DEBUG, v);
    }
  }
}
