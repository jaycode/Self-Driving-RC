/*
 * We can adjust throttle by writing analog values to IN1 and IN2 pins.
 * The values range from 81 to 255.
 */
const uint8_t engineENPin = 2;
const uint8_t engineIN1Pin = 3;
const uint8_t engineIN2Pin = 4;


void setup() {
  Serial.begin(9600);
  pinMode(engineIN1Pin, OUTPUT);
  pinMode(engineIN2Pin, OUTPUT);
  pinMode(engineENPin, OUTPUT);
  
  analogWrite(engineIN1Pin, 81);
//  analogWrite(engineIN1Pin, 255);
  digitalWrite(engineIN2Pin, LOW);
  digitalWrite(engineENPin, HIGH);
  delay(2000);
  digitalWrite(engineENPin, LOW);
}

void loop() {
//  controller.run();

}
