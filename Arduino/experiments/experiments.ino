// include our motor code
#include "motor.h"

// The Right Motors Enable Pin
// Labelled on the motor driver as ENA
// Be carful of PWM Timers
const int motorRENPin = 10;
// The Right Motors IN1 Pin
// Labelled on the motor driver as IN1
const int motorRIN2Pin = 9;
// The Right Motors IN2 Pin
// Labelled on the motor driver as IN2
const int motorRIN1Pin = 8;

// The Left Motors Enable Pin
// Labelled on the motor driver as ENB
// Be carful of PWM Timers
const int motorLENPin = 5;
// The Left Motors IN1 Pin
// Labelled on the motor driver as IN3
const int motorLIN2Pin = 7;
// The Left Motors IN2 Pin
// Labelled on the motor driver as IN4
const int motorLIN1Pin = 6;

int sensorPin = A0;    // select the input pin for the potentiometer
int sensorValue = 0;  // variable to store the value coming from the sensor
int targetPosition = 512; // 0 - 1024
int slack = 100;

// Create two Motor objects (instances of our Motor class)
// to drive each motor.
Motor rightMotor(motorRIN1Pin, motorRIN2Pin, motorRENPin);
Motor leftMotor(motorLIN1Pin, motorLIN2Pin, motorLENPin);

void setup()
{
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
}

void loop()
{ 
  // read the value from the sensor:
  sensorValue = analogRead(sensorPin);
  Serial.print(sensorValue);
  Serial.print("\n");
  
  leftMotor.freeRun();
  digitalWrite(LED_BUILTIN, HIGH);    // turn the LED off by making the voltage LOW
  delay(50);
  if (sensorValue - targetPosition > slack/2) {
    // Let's just make them go forward to test the
    // motors and the Motor class.
    //rightMotor.forward(100);
    leftMotor.forward(100); 
  } else if (sensorValue - targetPosition < -slack/2) {
    
    //rightMotor.backward(100);
    leftMotor.backward(100);
  } else {
    leftMotor.freeRun();
    leftMotor.brake();
  }
  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
  delay(10);
   
}

