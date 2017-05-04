/*
 * Header file for RC class (Remote Control).
 */

#ifndef _RC_H_
#define _RC_H_

// To get a value of a channel, use readValue(RC_CHANNEL_NAME)
// Where channel names are defined below.
// I used a single function to read channels to save memory.
// readValue uses switch cases instead of a hashmap since the former
// uses less memory.
#define RC_AUX1 0
#define RC_GEAR 1
#define RC_RUDO 2
#define RC_ELEV 3
#define RC_AILE 4
#define RC_THRO 5

#include <stdint.h>

class RC {
  public:
    RC(uint8_t pinAux1, uint8_t pinGear, uint8_t pinRudo,
       uint8_t pinElev, uint8_t pinAile, uint8_t pinThro);
    uint8_t readValue(uint8_t channel);
  protected:
    uint8_t normalize(float value, float valueMin, float valueMax,
                      float normMax);
  private:
    uint8_t pinAux1_;
    uint8_t pinGear_;
    uint8_t pinRudo_;
    uint8_t pinElev_;
    uint8_t pinAile_;
    uint8_t pinThro_;
};

#endif
