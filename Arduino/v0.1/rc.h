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
#include "helpers.h"

class RC {
  public:
    // set to -1 to disable a pin.
    RC(uint8_t pinAux1, uint8_t pinGear, uint8_t pinRudo,
       uint8_t pinElev, uint8_t pinAile, uint8_t pinThro);
    
    // Divides returned values based on defaultBins_ and defaultBinsSize_.
    int readDigital(const uint8_t channel, const int bins[], const int binsSize=3, const int tolerance=50);
    int readDigital(const uint8_t channel);
    
    // Read raw value.
    int readValue(uint8_t channel);

    int getMinA();
    int getMaxA();
    
  private:
    // Takes in value and B measurements, outputs the value in B range.
    int readNormalized(int value, int minB, int maxB);    

    int digitize(const int value, const int bins[], const int binsSize, const int tolerance=50);

    // Bins for digitize
    const int defaultBins_[3] = {1080, 1470, 1870};
    const int defaultBinsSize_ = 3;
    
    uint8_t pinAux1_;
    uint8_t pinGear_;
    uint8_t pinRudo_;
    uint8_t pinElev_;
    uint8_t pinAile_;
    uint8_t pinThro_;
};

#endif
