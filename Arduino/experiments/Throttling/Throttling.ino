/*
 * Figure out throttling forward and backward.
 */


const uint8_t engineIN1Pin =3;
const uint8_t engineIN2Pin = 9;
const uint8_t engineIN3Pin = 10;
const uint8_t engineIN4Pin = 4;

void setup() {
  Serial.begin(9600);
  pinMode(engineIN1Pin, OUTPUT);
  pinMode(engineIN2Pin, OUTPUT);
  pinMode(engineIN3Pin, OUTPUT);
  pinMode(engineIN4Pin, OUTPUT);
  
  // Forward
  analogWrite(engineIN1Pin, 90);
  analogWrite(engineIN2Pin, LOW);
  analogWrite(engineIN3Pin, LOW);
  analogWrite(engineIN4Pin, 90);

  // Backward
//  analogWrite(engineIN1Pin, LOW);
//  analogWrite(engineIN2Pin, 90);
//  analogWrite(engineIN3Pin, 90);
//  analogWrite(engineIN4Pin, LOW);

  delay(2000);
  analogWrite(engineIN1Pin, LOW);
  digitalWrite(engineIN2Pin, LOW);
  digitalWrite(engineIN3Pin, LOW);
  digitalWrite(engineIN4Pin, LOW);
}

void loop() {
//  controller.run();

}
