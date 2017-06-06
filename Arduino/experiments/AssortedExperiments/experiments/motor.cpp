/*
 * Class file for motor driver.
 * In this file we implement the methods
 * we described in the header file.
 * Refer to the Motor Driver Truth Table at the
 * "Understanding the Motor Driver" section.
 *
 * created 27 Sep 2013
 * by Blaise Jarrett
 *
 * This example code is in the public domain.
 *
 */

// Include our class header
#include "motor.h"
// Include the "Arduino" Library
// This is required to give us access to the pinMode,
// digitalWrite, and analogWrite functions
#include <Arduino.h>


// Constructor implementation
// Use the ":" shorthand for initializing class members
// This is called a initialization list
Motor::Motor(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinEN) : 
             pinIN1_(pinIN1), pinIN2_(pinIN2), pinEN_(pinEN)
{
  // set all the pins as outputs
  pinMode(pinIN1_, OUTPUT);
  pinMode(pinIN2_, OUTPUT);
  pinMode(pinEN_, OUTPUT);

  // disable the motor
  digitalWrite(pinEN_, LOW);
}

// The forward implementation
void Motor::forward(uint8_t speed)
{
  // set the control lines to drive the motor forward
  // This is described as "Direction B" in the
  // "Understanding the Motor Driver" truth table
  // (Row# 7)
  digitalWrite(pinIN1_, HIGH);
  digitalWrite(pinIN2_, LOW);

  // use the enable line with PWM to control the speed
  analogWrite(pinEN_, speed);
}

// The backward implementation
void Motor::backward(uint8_t speed)
{
  // set the control lines to drive the motor backwards
  // This is described as "Direction A" in the 
  // "Understanding the Motor Driver" truth table
  // (Row# 6)
  digitalWrite(pinIN1_, LOW);
  digitalWrite(pinIN2_, HIGH);

  // use the enable line with PWM to control the speed
  analogWrite(pinEN_, speed);
}

// the brake implementation
void Motor::brake()
{
  // set the control lines to brake the motor
  // This is described in the 
  // "Understanding the Motor Driver" truth table
  // (Row# 5)
  // We could have used either Row# 5 or #8. We
  // picked #5 over #8 by random choice.
  digitalWrite(pinIN1_, LOW);
  digitalWrite(pinIN2_, LOW);
  digitalWrite(pinEN_, HIGH);
}

// the free run implementation
void Motor::freeRun()
{
  // set the control lines to allow the motor to
  // free run. This is described in the 
  // "Understanding the Motor Driver" truth table.
  // Row# 1-4 all result in free runs. Any time the enable
  // pin is LOW we get a free run.
  // We can just set the enable pin LOW and don't have to
  // worry about the IN1 and IN2 pins.
  digitalWrite(pinEN_, LOW);
}
