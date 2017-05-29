#include "motor.h"
#include <Arduino.h>

extern  uint8_t engineENPin;
extern  uint8_t engineIN1Pin;
extern  uint8_t engineIN2Pin;

extern  uint8_t steerENPin;
extern  uint8_t steerIN1Pin;
extern  uint8_t steerIN2Pin;

Motor::Motor(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinEN) : 
             pinIN1_(pinIN1), pinIN2_(pinIN2), pinEN_(pinEN)
{
  // set all the pins as outputs
//  pinMode(pinIN1_, OUTPUT); //==engineIN2Pin
 // pinMode(pinIN2_, OUTPUT); //==engineIN1Pin
 // pinMode(pinEN_, OUTPUT);  //==engineENPin

  pinMode(engineIN2Pin, OUTPUT); //==engineIN2Pin
  pinMode(engineIN1Pin, OUTPUT); //==engineIN1Pin
  pinMode(engineENPin, OUTPUT);  //==engineENPin

  pinMode(steerIN1Pin, OUTPUT); //==engineIN2Pin
  pinMode(steerIN2Pin, OUTPUT); //==engineIN1Pin
  pinMode(steerENPin, OUTPUT);  //==engineENPin

  // disable the motor
  digitalWrite(engineENPin, HIGH);
  digitalWrite(steerENPin, HIGH);
}

void Motor::forward(int speed)
{
  /*
   * Spin the motor forward
   */
  speed = min(255,max(0,speed));
  
  analogWrite(engineIN1Pin, 0);
  analogWrite(engineIN2Pin, speed);
//  digitalWrite(engineENPin, HIGH);


  analogWrite(steerIN2Pin, 0);
  analogWrite(steerIN1Pin, speed);
//  digitalWrite(steerENPin, HIGH);
}

void Motor::backward(int speed)
{
  /*
   * Spin the motor backward
   */
  speed = min(255,max(0,speed));

  analogWrite(engineIN1Pin, speed);
  analogWrite(engineIN2Pin, 0);
  // use the enable line with PWM to control the speed
//  digitalWrite(engineENPin, HIGH);

  analogWrite(steerIN2Pin, speed);
  analogWrite(steerIN1Pin, 0);
  // use the enable line with PWM to control the speed
//  digitalWrite(steerENPin, HIGH);
}

void Motor::brake()
{
  /*
   * Braking
   */
 /*  
  digitalWrite(pinIN1_, LOW);
  digitalWrite(pinIN2_, LOW);
  digitalWrite(pinEN_, HIGH);
 */ 
}

void Motor::freeRun()
{
  /*
   * Let the motor free runs.
   */
   
//  digitalWrite(pinEN_, LOW);
}

SteeringWheel::SteeringWheel(
  uint8_t pinIN1, uint8_t pinIN2, uint8_t pinEN, uint8_t pinFeed) : 
  Motor(pinIN1, pinIN2, pinEN), pinFeed_(pinFeed){
  pinMode(pinFeed_, INPUT);
}

int SteeringWheel::readFeed() {
  return(analogRead(pinFeed_));
}

float SteeringWheel::readNormValue(float minB, float maxB) {
  return normalize((float)readFeed(), (float)minFeed_, (float)maxFeed_, minB, maxB);
}

