/*
 * Header file for motor driver class
 *
 * created 27 Sep 2013
 * by Blaise Jarrett
 *
 * This example code is in the public domain.
 *
 */

// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _MOTOR_H_
#define _MOTOR_H_

// Include standard ints so we can use specific integer
// types (uint8_t - unsigned 8 bits)
// See: http://www.nongnu.org/avr-libc/user-manual/group__avr__stdint.html
#include <stdint.h>

// Declare our motor class
// For info on Object Oriented Programming see:
// http://en.wikipedia.org/wiki/Object-oriented_programming
class Motor {
  // Access modifier - Make these members public
  // See: http://en.wikipedia.org/wiki/Access_modifiers
  public:
    // Constructor - Creates our Motor object from 3 pins
    Motor(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinEN);
    // Drive the motor forward at a defined speed
    // Speed is an 8bit unsigned integer. Max 255, Min 0.
    void forward(uint8_t speed);
    // Drive the motor backwards at a defined speed
    // Speed is an 8bit unsigned integer. Max 255, Min 0.
    void backward(uint8_t speed);
    // Brakes the motor
    void brake();
    // Lets the motor freely spin
    void freeRun();
  // Access modifier - Make these members private
  private:
    // We need instance variables to
    // remember the pins for the motor.
    uint8_t pinIN1_;
    uint8_t pinIN2_;
    uint8_t pinEN_;
// The end of our class declaration
};

// The end of our include guard
#endif
