/*
 * Header file for Car class. 
 */
// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _CAR_H_
#define _CAR_H_

#include <stdint.h>
#include "motor.h"
#include "rc.h"

class Car {
  public:
    Car(const Motor& inEngine, const Motor& inSteer, const RC& inRemoteController);
    void accelerateTo(uint8_t speed);
    void steerTo(float angle);
    void brake();
    uint8_t getSteerSpeed();
    void setSteerSpeed();
    

    // Print to Serial at baud rate given by logSerialBaudRate.
//    void log();
  private:
    uint8_t steerSpeed_ = 100;
    Motor engine_;
    Motor steer_;
    RC rc_;
    float curAngle_ = 0;
    int logSerialBaudRate_ = 9600;
};

#endif
