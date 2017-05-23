#include "car.h"
#include <Arduino.h>

Car::Car(const Motor& inEngine, const SteeringWheel& inSteer, const RC& inRemoteController) : 
         engine_(inEngine), steer_(inSteer), rc_(inRemoteController) {
}

void Car::accelerateTo(int speed) {
  if (engineReverse_) {
    // For when we mixed up the cables.
    speed = -speed;
  }
  if (speed > 0) {
    engine_.forward(speed);
  }
  else if (speed < 0) {
    engine_.backward(-speed);
  }
  else if (speed == 0) {
    engine_.brake();
  }
}

void Car::brake() {
  engine_.brake();
}

void Car::listen() {
  uint8_t newMode = rc_.readDigital(RC_AUX1);
  listenComputer();
  if (newMode != curDriveMode_) {
    sendCommand(CMD_CHANGE_DRIVE_MODE, newMode);
    curDriveMode_ = newMode;
  }
  // Reads current steering angle, needed in listenAile().
  curSteerFeed_ = steer_.readFeed();
  if (curDriveMode_ != DRIVE_MODE_AUTO) {
    // We set steer speed as 0 here to allow for two joysticks controlling the steering wheel.
    steerSpeed_ = 0;
    if (elevSpeed_ == 0) {
      listenThro(); // throttling left joystick
    }
    if (throSpeed_ == 0) {
      listenElev(); // throttling right joystick
    }
    listenAile(); // steering right joystick
    accelerateTo(throSpeed_+elevSpeed_);
    steer(steerSpeed_);
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
  ss = log10(fabs(ss))*110;
  
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

void Car::setCurDriveMode(uint8_t value) {
  curDriveMode_ = value;
}


void Car::listenComputer() {
  char ccmd = Serial.read();
  if (ccmd == CCMD_DRIVE_MODE) {
    sendCommand(CMD_CHANGE_DRIVE_MODE, curDriveMode_);
  }
  if (ccmd == CCMD_STEER) {
    sendCommand(CMD_STEER, curSteerFeed_);
  }
}

