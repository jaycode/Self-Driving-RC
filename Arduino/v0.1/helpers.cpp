#include "helpers.h"
#include "constants.h"

float normalize(float value, float minA, float maxA, float minB, float maxB) {
  /*
   * Takes in value and A and B measurements, outputs the value in B range.
   * 
   */
  float rangeA = maxA - minA;
  float rangeB = maxB - minB;
  float result = minB + ( ((value-minA)/rangeA) * rangeB);
  if (result < minB) {
    result = minB;
  }
  else if (result > maxB) {
    result = maxB;
  }
  return result;
}

  
float normalizeBi(float value, float minA, float maxA, float minB, float maxB) {
  /*
   * In normalizeBi (bi for Bidirectional), the center value should be 0.
   * If minB and maxB were set to 20 to 255, respectively, that means
   * the value range will be between -255 to 255, and values between -20 and 20 are
   * turned into 0.
   * 
   */
  float result = normalize(value, minA, maxA, -maxB, maxB);
  if (result < minB && result > - minB) {
    result = 0;
  }
  else if (result > maxB) {
    result = maxB;
  }
  else if (result < -maxB) {
    result = -maxB;
  }
  return result;
}

// Todo: Get this to work with templates
void sendCommand(char cmd, int val) {
  Serial.print(CMD_BEGIN);
  Serial.print(cmd);
  Serial.println(val);
}

void sendCommand(char cmd, unsigned int val) {
  Serial.print(CMD_BEGIN);
  Serial.print(cmd);
  Serial.println(val);
}

void sendCommand(char cmd, uint8_t val) {
  Serial.print(CMD_BEGIN);
  Serial.print(cmd);
  Serial.println(val);
}

void sendCommand(char cmd, char val[]) {
  Serial.print(CMD_BEGIN);
  Serial.print(cmd);
  Serial.println(val);
}

void sendCommand(char cmd, String val) {
  Serial.print(CMD_BEGIN);
  Serial.print(cmd);
  Serial.println(val);
}

unsigned int chars2int(char b[]) {
  return int((unsigned char)(b[0]) << 24 |
              (unsigned char)(b[1]) << 16 |
              (unsigned char)(b[2]) << 8 |
              (unsigned char)(b[3]));
}

