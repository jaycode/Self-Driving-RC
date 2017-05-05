/*
 * This example builds up from the Throttling example, to demonstrate
 * how the car may "listen" to inputs from remote controller while adjusting
 * its speed.
 * 
 */
#include <Thread.h>
#include <StaticThreadController.h>

enum States {STATE_ACTIVE, STATE_DELAY};

const uint8_t engineENPin = 2;
const uint8_t engineIN1Pin = 3;
const uint8_t engineIN2Pin = 4;

const uint8_t aux1Pin = 8;
const uint8_t rudoPin = 9;
const uint8_t elevPin = 10;
const uint8_t ailePin = 11;
const uint8_t throPin = 12;

Thread engineThread = Thread();
Thread rcThread = Thread();
StaticThreadController<2> controller (&engineThread, &rcThread);

int maxActive = 20;
int activeCycle = 0;
int maxDelay = 50;
int delayCycle = 50;

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

  rcThread.setInterval(1);
//  rc
}

void loop() {
//  controller.run();

}
