/*
 * We can adjust throttle by writing analog values to IN1 and IN2 pins.
 * The values range from 81 to 255.
 */
const uint8_t engineENPin = 7;
const uint8_t engineIN1Pin = 5;
const uint8_t engineIN2Pin = 6;
const uint8_t steerFeedPin = A0;


void setup() {
  Serial.begin(9600);
  pinMode(engineIN1Pin, OUTPUT);
  pinMode(engineIN2Pin, OUTPUT);
  pinMode(engineENPin, OUTPUT);
  pinMode(steerFeedPin, INPUT);
  
  analogWrite(engineIN1Pin, 34);
  digitalWrite(engineIN2Pin, LOW);
  digitalWrite(engineENPin, HIGH);
  delay(500);
  digitalWrite(engineENPin, LOW);
}

void loop() {
  Serial.println(analogRead(steerFeedPin));
}
