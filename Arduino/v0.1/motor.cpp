#include "motor.h"
#include <Arduino.h>

Motor::Motor(uint8_t pinIN1, uint8_t pinIN2, uint8_t pinIN3, uint8_t pinIN4) : 
             pinIN1_(pinIN1), pinIN2_(pinIN2), pinIN3_(pinIN3), pinIN4_(pinIN4)
{
  // set all the pins as outputs
  pinMode(pinIN1_, OUTPUT);
  pinMode(pinIN2_, OUTPUT);
  if (pinIN3_) {
    pinMode(pinIN3_, OUTPUT);
    pinMode(pinIN4_, OUTPUT); 
  }
}

void Motor::forward(int speed)
{
  /*
   * Spin the motor forward
   */
  speed = min(255,max(0,speed));
  analogWrite(pinIN1_, speed);
  digitalWrite(pinIN2_, LOW);
  if (pinIN3_) {
    digitalWrite(pinIN3_, LOW);
    analogWrite(pinIN4_, speed);
  }
}

void Motor::backward(int speed)
{
  /*
   * Spin the motor backward
   */
  speed = min(255,max(0,speed));
  digitalWrite(pinIN1_, LOW);
  analogWrite(pinIN2_, speed);
  if (pinIN3_) {
    analogWrite(pinIN3_, speed);
    digitalWrite(pinIN4_, LOW);
  }
}

void Motor::brake()
{
  /*
   * Braking
   */
  digitalWrite(pinIN1_, LOW);
  digitalWrite(pinIN2_, LOW);
  if (pinIN3_) {
    digitalWrite(pinIN3_, LOW);
    digitalWrite(pinIN4_, LOW);
  }
}
  

SteeringWheel::SteeringWheel(
  uint8_t pinIN1, uint8_t pinIN2, uint8_t pinFeed) : 
  Motor(pinIN1, pinIN2, NULL, NULL), pinFeed_(pinFeed){
  pinMode(pinFeed_, INPUT);
}

int SteeringWheel::readFeed() {
  return(analogRead(pinFeed_));
}

float SteeringWheel::readNormValue(float minB, float maxB) {
  return normalize((float)readFeed(), (float)minFeed_, (float)maxFeed_, minB, maxB);
}

