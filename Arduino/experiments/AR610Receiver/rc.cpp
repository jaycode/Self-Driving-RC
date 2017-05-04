#include "rc.h"
#include <Arduino.h>

float maxRudo = 1600;
float minRudo = 1600;
float maxElev = 1600;
float minElev = 1600;
float maxAile = 1600;
float minAile = 1600;
float maxThro = 1600;
float minThro = 1600;

RC::RC(uint8_t pinAux1, uint8_t pinGear, uint8_t pinRudo, 
       uint8_t pinElev, uint8_t pinAile, uint8_t pinThro) : 
       pinAux1_(pinAux1), pinGear_(pinGear), pinRudo_(pinRudo),
       pinElev_(pinElev), pinAile_(pinAile), pinThro_(pinThro) {
  pinMode(pinAux1_, INPUT);
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
    case RC_AUX1: valueRaw = pulseIn (pinAux1_,HIGH);
                  Serial.print("ax: ");
                  Serial.print(valueRaw);
                  Serial.print(" ");
                  Serial.print(analogRead(pinAux1_));
                  break;
    case RC_GEAR: valueRaw = pulseIn (pinGear_,HIGH);
                  Serial.print(" g: ");
                  Serial.print(valueRaw);
                  Serial.print(" ");
                  Serial.print(analogRead(pinGear_));
                  break;
    case RC_RUDO: valueRaw = pulseIn (pinRudo_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" r: ");
//                  Serial.print(valueRaw);
                  if (valueRaw > maxRudo) {
                    maxRudo = valueRaw;
                  }
                  else if (valueRaw < minRudo) {
                    minRudo = valueRaw;
                  }
                  Serial.print(minRudo);
                  Serial.print(" ");
                  Serial.print(maxRudo);
                  break;
    case RC_ELEV: valueRaw = pulseIn (pinElev_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" e: ");
//                  Serial.print(valueRaw);
                  if (valueRaw > maxElev) {
                    maxElev = valueRaw;
                  }
                  else if (valueRaw < minElev) {
                    minElev = valueRaw;
                  }
                  Serial.print(minElev);
                  Serial.print(" ");
                  Serial.print(maxElev);
                  break;
    case RC_AILE: valueRaw = pulseIn (pinAile_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" a: ");
//                  Serial.print(valueRaw);
                  if (valueRaw > maxAile) {
                    maxAile = valueRaw;
                  }
                  else if (valueRaw < minAile) {
                    minAile = valueRaw;
                  }
                  Serial.print(minAile);
                  Serial.print(" ");
                  Serial.print(maxAile);
                  break;
    case RC_THRO: valueRaw = pulseIn (pinThro_,HIGH);
//                  normalize(valueRaw);
                  Serial.print(" t: ");
//                  Serial.println(valueRaw);
                  if (valueRaw > maxThro) {
                    maxThro = valueRaw;
                  }
                  else if (valueRaw < minThro) {
                    minThro = valueRaw;
                  }
                  Serial.print(minThro);
                  Serial.print(" ");
                  Serial.println(maxThro);
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

