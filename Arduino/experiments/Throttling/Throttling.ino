/*
 * Attempted to use multithreading with throttling. Turns out I was wrong. Do not use this project.
 */

#include <Thread.h>
#include <StaticThreadController.h>

enum States {STATE_ACTIVE, STATE_DELAY};

const uint8_t engineENPin = 2;
const uint8_t engineIN1Pin = 3;
const uint8_t engineIN2Pin = 4;

Thread engineThread = Thread();
Thread steerThread = Thread();
StaticThreadController<2> controller (&engineThread, &steerThread);

int maxActive = 1;
int activeCycle = 0;
int maxDelay = 10;
int delayCycle = 10;

States state = STATE_DELAY;

void engineCallback() {
  Serial.println("engineCallback");
  if (state == STATE_ACTIVE) {
    ++activeCycle;
    if (activeCycle > maxActive) {
      // Starts Delay cycle, stops engine.
      activeCycle = 0;
      state = STATE_DELAY;
      digitalWrite(engineENPin, LOW);
    }
  }
  else if (state == STATE_DELAY) {
    ++delayCycle;
    if (delayCycle > maxDelay) {
      // Starts active cycle, fires up engine.
      delayCycle = 0;
      state = STATE_ACTIVE;
      digitalWrite(engineIN1Pin, HIGH);
      digitalWrite(engineIN2Pin, LOW);
      analogWrite(engineENPin, 128);
    }
  }
}

void setup() {
  Serial.begin(9600);
  pinMode(engineIN1Pin, OUTPUT);
  pinMode(engineIN2Pin, OUTPUT);
  pinMode(engineENPin, OUTPUT);
  
  engineThread.setInterval(10);
  engineThread.enabled = true;
  engineThread.onRun(engineCallback);

//// This is the non-threaded version. With multithreading we basically
//// emulate this, while having another thread to listen to other sensors and
//// inputs.
//
//  digitalWrite(engineIN1Pin, HIGH);
//  digitalWrite(engineIN2Pin, LOW);
//  for (int i=0; i<10; ++i) {
//    analogWrite(engineENPin, 128);
//    delay(10);
//    digitalWrite(engineENPin, LOW);
//    delay(100);
//  }
//  delay(100);

//// There is no difference between setting the EN pin to 128 or 200:
//  digitalWrite(engineIN1Pin, HIGH);
//  digitalWrite(engineIN2Pin, LOW);
  analogWrite(engineIN1Pin, 80);
  digitalWrite(engineIN2Pin, LOW);
//  analogWrite(engineENPin, 128);
  digitalWrite(engineENPin, HIGH);
  delay(2000);
  digitalWrite(engineENPin, LOW);
}

void loop() {
//  controller.run();

}
