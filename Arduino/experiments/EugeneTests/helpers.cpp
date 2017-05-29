#include "helpers.h"
#include <Arduino.h>

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


