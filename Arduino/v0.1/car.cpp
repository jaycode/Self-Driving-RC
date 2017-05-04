#include "car.h"
#include <Arduino.h>

Car::Car(const Motor& inEngine, const Motor& inSteer, const RC& inRemoteController) : 
         engine_(inEngine), steer_(inSteer), rc_(inRemoteController) {
  Serial.begin(logSerialBaudRate_);
}

void Car::accelerateTo(uint8_t speed) {
  if (speed > 0) {
    engine_.forward(speed);
  }
  else if (speed < 0) {
    engine_.backward(-speed);
  }
  else if (speed == 0) {
    engine_.freeRun();
  }
}

void Car::steerTo(float angle) {
  float angleDiff = (angle - curAngle_)*100;
  if (angleDiff < 0) {
    steer_.backward(steerSpeed_);
  }
  else {
    steer_.forward(steerSpeed_);
  }
  uint8_t angleDiffInt = uint8_t(angleDiff);
  delay(abs(angleDiffInt));
  steer_.brake();
}

void Car::brake() {
  engine_.brake();
}

//void Car::log() {
//  Serial.print("gear: ");
//  Serial.print(rc.getValue(RC::Channels.GEAR));
//  Serial.print(" rudo: ");
//  Serial.print(rc.getValue(RC::Channels.RUDO));
//  Serial.print(" elev: ");
//  Serial.print(rc.getValue(RC::Channels.ELEV));
//  Serial.print("  aile: ");
//  Serial.print(rc.getValue(RC::Channels.AILE));
//  Serial.print("  throttle: ");
//  Serial.println(rc.getValue(RC::Channels.THRO));
//}

