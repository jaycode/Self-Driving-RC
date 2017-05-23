/*
 * Microcontroller side
 */
 
#include <stdint.h>

// Look into constants.h and ensure all constants have their counterparts
// in the computer side.
#include "constants.h"

#include "motor.h"
#include "car.h"
#include "rc.h"
#include "helpers.h"

uint8_t engineIN1Pin = 3;
uint8_t engineIN2Pin = 9;
uint8_t engineIN3Pin = 10;
uint8_t engineIN4Pin = 11;
 
uint8_t steerIN1Pin = 5; // 1A1
uint8_t steerIN2Pin = 6; // 1B1
const uint8_t steerFeedPin = A0;

const uint8_t aux1Pin = 2;
const uint8_t gearPin = 4;
const uint8_t rudoPin = 7;
const uint8_t elevPin = 8;
const uint8_t ailePin = 12;
const uint8_t throPin = 13;

Motor engine(engineIN1Pin, engineIN2Pin, engineIN3Pin, engineIN4Pin);
SteeringWheel steer(steerIN1Pin, steerIN2Pin, steerFeedPin);
RC rc(aux1Pin, gearPin, rudoPin, elevPin, ailePin, throPin);
  
Car car(engine, steer, rc);

void setup() {
  Serial.begin(9600);
  uint8_t aux1 = rc.readDigital(RC_AUX1);
  car.setCurDriveMode(aux1);
  sendCommand(CMD_CHANGE_DRIVE_MODE, aux1);
}

void loop() {
  car.listen();
}
