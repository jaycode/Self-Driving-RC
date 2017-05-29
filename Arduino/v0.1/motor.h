/*
 * Header file for motor driver class
 */

// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _MOTOR_H_
#define _MOTOR_H_

// Include standard ints so we can use specific integer
// types (uint8_t - unsigned 8 bits)
// See: http://www.nongnu.org/avr-libc/user-manual/group__avr__stdint.html
#include <stdint.h>
#include "helpers.h"

class Motor {
  public:
    // Constructor - Creates our Motor object from 4 pins
    Motor(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinIN3, uint8_t pinIN4);
    // Drive the motor forward at a defined speed
    // Speed is an 8bit unsigned integer. Max 255, Min 0.
    void forward(int speed);
    // Drive the motor backwards at a defined speed
    // Speed is an 8bit unsigned integer. Max 255, Min 0.
    void backward(int speed);
    // Brakes the motor
    void brake();
    // Lets the motor freely spin
    void freeRun();
    
  protected:
    uint8_t pinIN1_;
    uint8_t pinIN2_;
    uint8_t pinIN3_;
    uint8_t pinIN4_;
};

class SteeringWheel : public Motor {
  /*
   * Motor + feed pin.
   */
  public:
    SteeringWheel(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinFeed);
    int readFeed();
    float readNormValue(float minB, float maxB); 
  protected:
    uint8_t pinFeed_;
    int minFeed_ = 0;
    int maxFeed_ = 1023;
};

#endif
