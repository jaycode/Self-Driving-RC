#include <stdint.h>
#include "motor.h"
#include "car.h"
#include "rc.h"

const uint8_t engineENPin = 2;
const uint8_t engineIN1Pin = 3;
const uint8_t engineIN2Pin = 4;

const uint8_t steerENPin = 7;
const uint8_t steerIN1Pin = 5;
const uint8_t steerIN2Pin = 6;
const uint8_t steerFeedPin = 13;

const uint8_t gearPin = 12;
const uint8_t rudoPin = 11;
const uint8_t elevPin = 10;
const uint8_t ailePin = 9;
const uint8_t throPin = 8;

const double rudoMin = 1093;
const double rudoMax = 1891;

const double throSlack = 200;


const double slack = 30;



Motor engine(engineIN2Pin, engineIN1Pin, engineENPin);
Motor steer(steerIN1Pin, steerIN2Pin, steerENPin);
RC rc(gearPin, rudoPin, elevPin, ailePin, throPin);
  
Car car(engine, steer, rc);
//car.setup(
//  engineMin = 130;
//  engineMax = 200;
//  steerMin
//)

void setup() {
  Serial.begin(9600);
//  car.steerTo(0); // Steer X degrees to the right.
//  car.accelerateTo(80);
//  delay(2000);
//  car.brake();

  // Test steering and throttling
////  steer.forward(60);
//  engine.forward(127);
//  delay(1000);
////  steer.freeRun();
//  engine.freeRun();
//  delay(5000);
////  steer.backward(60);
//  engine.backward(127);
//  delay(1000);
//  steer.freeRun();
//  engine.freeRun();
}

void report() {
  /* Report all pins inputs
  */  
  
}

void loop() {
  // put your main code here, to run repeatedly:
  uint8_t gear = rc.readValue(RC_GEAR);
  uint8_t rudo = rc.readValue(RC_RUDO);
  uint8_t elev = rc.readValue(RC_ELEV);
  uint8_t aile = rc.readValue(RC_AILE);
  uint8_t thro = rc.readValue(RC_THRO);
//  steer.backward(200);
//  engine.forward(200);
//  delay(1000);
//  steer.brake();
//  engine.freeRun();
//  delay(5000);
  
//  double rudo = pulseIn (inPinRudo,HIGH);  //Read and store channel 1
//  double rudoN = rudoNorm(rudo);
//  Serial.print ("Rudo:");  //Display text string on Serial Monitor to distinguish variables
//  Serial.print (rudoN);
//  Serial.println ("");
//  
//  if (rudoN > slack/2) {
//    steer.forward(rudoN);
//  } else if (rudoN < -slack/2){
//    steer.backward(-rudoN);
//  } else {
//    steer.freeRun();
//    steer.brake();
//  }
//
//  double thro = pulseIn (inPinThro,HIGH);  //Read and store channel 1
//  double throN = throNorm(thro);
//  Serial.print ("Thro:");  //Display text string on Serial Monitor to distinguish variables
//  Serial.print (throN);
//  Serial.println ("");
//  
//  if (throN > throSlack/2) {
//    engine.forward(throN);
//  } else if (throN < -throSlack/2){
//    engine.backward(-throN);
//  } else {
//    engine.freeRun();
//    engine.brake();
//  }
}
