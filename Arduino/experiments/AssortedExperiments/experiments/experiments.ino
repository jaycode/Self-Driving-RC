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

//Define the aggressive and conservative Tuning Parameters
double aggKp=4, aggKi=0.2, aggKd=1;
double consKp=1, consKi=0.05, consKd=0.25;

//Specify the links and initial tuning parameters
PID myPID(&Input, &Output, &Setpoint, consKp, consKi, consKd, DIRECT);


void setup()
{
  Serial.begin(9600);
  pinMode(LED_BUILTIN, OUTPUT);
  //initialize the variables we're linked to
  Input = analogRead(sensorPin);
  Setpoint = 512;

  //turn the PID on
  myPID.SetMode(AUTOMATIC);
}

void loop()
{ 
  // read the value from the sensor:
  Input = analogRead(sensorPin);
  Serial.print("Input: ");
  Serial.print(Input);
  Serial.print("\n");

  double gap = abs(Setpoint-Input); //distance away from setpoint
  if (gap < 100)
  {  //we're close to setpoint, use conservative tuning parameters
    myPID.SetTunings(consKp, consKi, consKd);
  }
  else
  {
     //we're far from setpoint, use aggressive tuning parameters
     myPID.SetTunings(aggKp, aggKi, aggKd);
  }

  myPID.SetTunings(consKp, consKi, consKd);
  
  myPID.Compute();
  Output -= 125;
  if (abs(Output) < 50) {
    leftMotor.freeRun();
    leftMotor.brake();
  } else {
    if (Output > 0)  
      leftMotor.backward(Output);
    else
      leftMotor.forward(-Output);
  }

  Serial.print("Output: ");
  Serial.print(Output);
  Serial.print("\n");

  digitalWrite(LED_BUILTIN, HIGH);    // turn the LED off by making the voltage LOW
  delay(50);
  leftMotor.freeRun();
  leftMotor.brake();
    //leftMotor.brake();
  //delay(500);
//  }
//  digitalWrite(LED_BUILTIN, LOW);    // turn the LED off by making the voltage LOW
//  delay(10);
   
}

