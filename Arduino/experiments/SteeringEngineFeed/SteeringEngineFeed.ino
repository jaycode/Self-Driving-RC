// include our motor code
#include "motor.h"
#include <PID_v1.h>

// The Right Motors Enable Pin
// Labelled on the motor driver as ENA
// Be carful of PWM Timers
const int motorRENPin = 2;
// The Right Motors IN1 Pin
// Labelled on the motor driver as IN1
const int motorRIN2Pin = 4;
// The Right Motors IN2 Pin
// Labelled on the motor driver as IN2
const int motorRIN1Pin = 3;

// The Left Motors Enable Pin
// Labelled on the motor driver as ENB
// Be carful of PWM Timers
const int motorLENPin = 7;
// The Left Motors IN1 Pin
// Labelled on the motor driver as IN3
const int motorLIN2Pin = 6;
// The Left Motors IN2 Pin
// Labelled on the motor driver as IN4
const int motorLIN1Pin = 5;

int sensorPin = A0;    // select the input pin for the potentiometer
//int sensorValue = 0;  // variable to store the value coming from the sensor

// Create two Motor objects (instances of our Motor class)
// to drive each motor.
Motor rightMotor(motorRIN1Pin, motorRIN2Pin, motorRENPin);
Motor leftMotor(motorLIN1Pin, motorLIN2Pin, motorLENPin);

//Define Variables we'll be connecting to
double Setpoint, Input, Output;

void setup()
{
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  //initialize the variables we're linked to
  Input = analogRead(sensorPin);
  Setpoint = 512;
}

void loop()
{ 
  // read the value from the sensor:
  Input = analogRead(sensorPin);
  Serial.print("Input: ");
  Serial.print(Input);
  Serial.print("\n");
  digitalWrite(LED_BUILTIN, HIGH);    // turn the LED off by making the voltage LOW   
}

