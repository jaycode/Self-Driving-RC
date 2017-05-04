#include "rc.h"
#include <Arduino.h>

RC::RC(uint8_t pinGear, uint8_t pinRudo, uint8_t pinElev,
       uint8_t pinAile, uint8_t pinThro) : 
       pinGear_(pinGear), pinRudo_(pinRudo), pinElev_(pinElev),
       pinAile_(pinAile), pinThro_(pinThro) {
  pinMode(pinGear_, INPUT);
  pinMode(pinRudo_, INPUT);
  pinMode(pinElev_, INPUT);
  pinMode(pinAile_, INPUT);
  pinMode(pinThro_, INPUT);
}

uint8_t RC::readValue(uint8_t channel) {
  float valueRaw;
  uint8_t value;
  switch(channel) {
    case RC_GEAR: valueRaw = pulseIn (pinGear_,HIGH);
//                  normalize(valueRaw);
                  Serial.print("gear: ");
                  Serial.print(valueRaw);
                  break;
    case RC_RUDO: valueRaw = pulseIn (pinRudo_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" rudo: ");
                  Serial.print(valueRaw);
                  break;
    case RC_ELEV: valueRaw = pulseIn (pinElev_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" elev: ");
                  Serial.print(valueRaw);
                  break;
    case RC_AILE: valueRaw = pulseIn (pinAile_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" aile: ");
                  Serial.print(valueRaw);
                  break;
    case RC_THRO: valueRaw = pulseIn (pinThro_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" thro: ");
                  Serial.println(valueRaw);
                  break;
  }
  return 0;
}

uint8_t RC::normalize(float value, float valueMin, float valueMax,
                      float normMax) {
  float output = (((value - valueMin)/(valueMax - valueMin)) * (normMax*2)) - normMax;
  if (output < -normMax) {
    output = -normMax;
  }
  else if (output > normMax) {
    output = normMax;
  }
  return output;
}

