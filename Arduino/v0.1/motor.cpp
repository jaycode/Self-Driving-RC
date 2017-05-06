#include "motor.h"
#include <Arduino.h>

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

void Motor::forward(int speed)
{
  /*
   * Spin the motor forward
   */
  speed = min(255,max(0,speed));
  analogWrite(pinIN1_, speed);
  Serial.println(speed);
  digitalWrite(pinIN2_, LOW);

  digitalWrite(pinEN_, HIGH);
}

void Motor::backward(int speed)
{
  /*
   * Spin the motor backward
   */
  speed = min(255,max(0,speed));
  digitalWrite(pinIN1_, LOW);
  analogWrite(pinIN2_, speed);
  Serial.println(speed);

  // use the enable line with PWM to control the speed
  digitalWrite(pinEN_, HIGH);
}

void Motor::brake()
{
  /*
   * Braking
   */
  digitalWrite(pinIN1_, LOW);
  digitalWrite(pinIN2_, LOW);
  digitalWrite(pinEN_, HIGH);
}

void Motor::freeRun()
{
  /*
   * Let the motor free runs.
   */
  digitalWrite(pinEN_, LOW);
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

