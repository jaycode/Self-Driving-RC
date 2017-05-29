#include <stdint.h>
#include "motor.h"
#include "car.h"
#include "rc.h"

 uint8_t engineENPin = 2;
 uint8_t engineIN1Pin = 3;
 uint8_t engineIN2Pin = 11;

 uint8_t steerENPin = 7;
 uint8_t steerIN1Pin = 5;
 uint8_t steerIN2Pin = 6;
const uint8_t steerFeedPin = A0;

const uint8_t gearPin = -1; //unused
const uint8_t aux1Pin = 8;
const uint8_t rudoPin = 9;
const uint8_t elevPin = 10;
const uint8_t ailePin = 12;
const uint8_t throPin = 13;


Motor engine(engineIN2Pin, engineIN1Pin, engineENPin);
SteeringWheel steer(steerIN1Pin, steerIN2Pin, steerENPin, steerFeedPin);
RC rc(aux1Pin, gearPin, rudoPin, elevPin, ailePin, throPin);
  
Car car(engine, steer, rc);

void setup() {
  Serial.begin(9600);
}

void loop() {
  car.listen();
}
