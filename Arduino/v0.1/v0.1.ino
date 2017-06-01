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

uint8_t engineIN1Pin = 6;
uint8_t engineIN2Pin = 9;
//uint8_t engineIN3Pin = 10;
//uint8_t engineIN4Pin = 11;
 
uint8_t steerIN1Pin = 3; // 1A1
uint8_t steerIN2Pin = 5; // 1B1
const uint8_t steerFeedPin = A0;

const uint8_t aux1Pin = 11;
const uint8_t gearPin = 10;
const uint8_t rudoPin = 8;
const uint8_t elevPin = 7;
const uint8_t ailePin = 4;
const uint8_t throPin = 2;

//Motor engine(engineIN1Pin, engineIN2Pin, engineIN3Pin, engineIN4Pin);
Motor engine(engineIN1Pin, engineIN2Pin, 0, 0);
SteeringWheel steer(steerIN1Pin, steerIN2Pin, steerFeedPin);
RC rc(aux1Pin, gearPin, rudoPin, elevPin, ailePin, throPin);
  
Car car(engine, steer, rc);

void setup() {
  Serial.begin(9600);
//  Serial.setTimeout(200);
  uint8_t aux1 = rc.readDigital(RC_AUX1);
  car.setCurDriveMode(aux1);
  sendCommand(CMD_CHANGE_DRIVE_MODE, aux1);
}

void loop() {
  car.listen();
}
