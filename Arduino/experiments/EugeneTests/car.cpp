#include "car.h"
#include <Arduino.h>

Car::Car(const Motor& inEngine, const SteeringWheel& inSteer, const RC& inRemoteController) : 
         engine_(inEngine), steer_(inSteer), rc_(inRemoteController) {
}

void Car::accelerateTo(int speed) {
  if (engineReverse_) {
    speed = -speed;
  }

  if (speed > 0)
    engine_.forward(speed);
  else
    engine_.backward(-speed);
      
/*  
  if (speed > 0) {
    braked_ = false;
    engine_.forward(speed);
  }
  else if (speed < 0) {
    braked_ = false;
    engine_.backward(-speed);
  }
  else if (speed == 0) {
    if (braked_) {
      engine_.freeRun();
    }
    else {
      engine_.brake();
      braked_ = true;
    }
  }
*/  
}

void Car::brake() {
  engine_.brake();
}

void Car::listen() {
  // We set steer speed as 0 here to allow for two joysticks controlling the steering wheel.
  steerSpeed_ = 0;
  if (elevSpeed_ == 0) {
//    Serial.print("listen thro ");
    listenThro();
  }
  if (throSpeed_ == 0) {
    listenElev();
  }
  curSteerAngle_ = steer_.readNormValue(steerAngleMin_, steerAngleMax_);
  listenAile();
//  syncSteering(steerSpeed_);
  accelerateTo(throSpeed_+elevSpeed_);
  steer(steerSpeed_);
}

void Car::listenThro() {
  /**
   * The car can only move forward with throttle (due to how the remote control
   * was designed).
   */
  throSpeed_ = rc_.readNormValue(RC_THRO, engineMin_, engineMax_);
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
    elevSpeed_ = rc_.readNormValue(RC_ELEV, engineMin_, engineMax_);
    if (elevSpeed_ == engineMin_) {
      elevSpeed_ = 0;
    }
  }
}

void Car::listenAile() {
  int aile = rc_.readValue(RC_AILE);
  if (aile == 0) {
    // Means the remote controller is off.
    curRCAngle_ = 0;
    steerSpeed_ = 0;
  }
  else {
    curRCAngle_ = normalizeBi(aile, rc_.getMinA(), rc_.getMaxA(), steerAngleMin_, steerAngleMax_);
    steerSpeed_ += normalizeBi(curRCAngle_, steerAngleMin_, steerAngleMax_, steerMin_, steerMax_);
  }
}

void Car::steer(const int steerSpeed) {
  return;

  
  int ss = steerSpeed;
  ss = log10(fabs(ss))*100;
  
  if (steerSpeed < 0) {
    ss = -ss;
  }
  if (steerReverse_) {
    ss = -ss;
  }

  if (ss > 0) {
    if (curSteerAngle_ < steerAngleMin_+steerSlack_) {
      steer_.brake();
    }
    else {
      steer_.forward(ss);
    }
  }
  else if (ss < 0) {
    if (curSteerAngle_ > steerAngleMax_-steerSlack_) {
      steer_.brake();
    }
    else {
      steer_.backward(-ss);
    }
  }
  else {
    steer_.freeRun();
  }
}

void Car::syncSteering(const int steerSpeed) {
  /*
   * Synchronize current RC's and car wheels' steering angles.
   * 
   */
  float diff = curRCAngle_ - curSteerAngle_;
  int ss = (steerSpeed/100)*(steerSpeed/100);
  if (steerReverse_) {
    diff = -diff;
  }
  if (diff > steerSlack_) {
    steer_.forward(ss);
  }
  else if (diff < -steerSlack_) {
    steer_.backward(ss);
  }
  else {
    steer_.freeRun();
  }
}
