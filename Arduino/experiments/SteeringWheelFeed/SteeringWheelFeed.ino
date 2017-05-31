/*
 * Read feed from steering wheel and write to serial.
 * See if Python experiment SerialTest.py can read them.
 */
#include <stdint.h>

// Cannot use #define since it is a prepocessor.
const char CMD_BEGIN = 'C';
const char CMD_STEER = 'S';

int sensorPin = A0;    // select the input pin for the potentiometer

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
  Serial.print(CMD_BEGIN);
  Serial.print(CMD_STEER);
  Serial.println(Input);
  digitalWrite(LED_BUILTIN, HIGH);    // turn the LED off by making the voltage LOW   
}

