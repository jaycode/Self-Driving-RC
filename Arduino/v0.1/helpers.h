#ifndef _HELPERS_H_
#define _HELPERS_H_

#include "constants.h"
#include <Arduino.h>

float normalize(float value, float minA, float maxA, float minB, float maxB);
float normalizeBi(float value, float minA, float maxA, float minB, float maxB);

// Todo: Get this to work with templates, currently got error
// undefined reference to `void sendCommand<unsigned char>(char, unsigned char)'
// template<class T> void sendCommand(char cmd, T val);
//void sendCommand(char cmd, int val);
//void sendCommand(char cmd, unsigned int val);
//void sendCommand(char cmd, uint8_t val);
void sendCommand(char cmd, char val);
void sendCommand(char cmd, char val[]);
void sendCommand(char cmd, String val);

#endif
