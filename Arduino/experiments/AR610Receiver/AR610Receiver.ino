#include <stdint.h>
#include "rc.h"
const uint8_t aux1Pin = 13;
const uint8_t gearPin = 12;
const uint8_t rudoPin = 11;
const uint8_t elevPin = 10;
const uint8_t ailePin = 9;
const uint8_t throPin = 8;
RC rc(aux1Pin, gearPin, rudoPin, elevPin, ailePin, throPin);

void setup() {
  Serial.begin(9600);
}

void loop() {
  uint8_t aux1 = rc.readValue(RC_AUX1);
  uint8_t gear = rc.readValue(RC_GEAR);
  uint8_t rudo = rc.readValue(RC_RUDO);
  uint8_t elev = rc.readValue(RC_ELEV);
  uint8_t aile = rc.readValue(RC_AILE);
  uint8_t thro = rc.readValue(RC_THRO);
}
