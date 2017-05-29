#include "rc.h"
#include <Arduino.h>

RC::RC(uint8_t pinAux1, uint8_t pinGear, uint8_t pinRudo,
       uint8_t pinElev, uint8_t pinAile, uint8_t pinThro) : 
       pinAux1_(pinAux1), pinGear_(pinGear), pinRudo_(pinRudo),
       pinElev_(pinElev), pinAile_(pinAile), pinThro_(pinThro) {
  if (pinAux1_ > -1) {
    pinMode(pinAux1_, INPUT);
  }
  if (pinGear_ > -1) {
    pinMode(pinGear_, INPUT);
  }
  if (pinRudo_ > -1) {
    pinMode(pinRudo_, INPUT);
  }
  if (pinElev_ > -1) {
    pinMode(pinElev_, INPUT);
  }
  if (pinAile_ > -1) {
    pinMode(pinAile_, INPUT);
  }
  if (pinThro_ > -1) {
    pinMode(pinThro_, INPUT);
  }
}

int RC::readValue(uint8_t channel) {
  int valueRaw;
  switch(channel) {
    case RC_AUX1: valueRaw = pulseIn (pinAux1_,HIGH);
                  break;
    case RC_GEAR: valueRaw = pulseIn (pinGear_,HIGH);
                  break;
    case RC_RUDO: valueRaw = pulseIn (pinRudo_,HIGH);
                  break;
    case RC_ELEV: valueRaw = pulseIn (pinElev_,HIGH);
                  break;
    case RC_AILE: valueRaw = pulseIn (pinAile_,HIGH);
                  break;
    case RC_THRO: valueRaw = pulseIn (pinThro_,HIGH);
                  break;
  }
  return valueRaw;
}

int RC::readDigital(uint8_t channel, const int bins[]/*={1080, 1470, 1870}*/, const int binsSize, const int tolerance) {
  int valueRaw;
  int value;
  switch(channel) {
    case RC_AUX1: valueRaw = pulseIn (pinAux1_,HIGH);
                  value = digitize(valueRaw, bins, binsSize, tolerance);
                  break;
    case RC_GEAR: valueRaw = pulseIn (pinGear_,HIGH);
                  value = digitize(valueRaw, bins, binsSize, tolerance);
                  break;
  }
  return value;
}

int RC::readDigital(uint8_t channel) {
  int bins[] = {1080, 1470, 1870};
  return RC::readDigital(channel, bins);
}

int RC::digitize(const int value, const int bins[], const int binsSize, const int tolerance) {
  // C++ decays the array when it is passed into a function thus finding
  // size with the following methods won't work:
  // >>> int binsSize = sizeof(bins)/sizeof(bins[0]);
  // >>> int binsSize = sizeof(bins)/sizeof(int);
  // We pass binsSize parameter instead.
  for (int i=0; i<binsSize; i++) {
    int diff = value - bins[i];
    if (fabs(diff) <= tolerance) {
      return i;
    }
  }
  return (sizeof(bins)-1);
}

