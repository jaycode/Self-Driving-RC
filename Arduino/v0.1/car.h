/*
 * Header file for Car class. 
 */
// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _CAR_H_
#define _CAR_H_

#include <stdint.h>
#include "motor.h"
#include "rc.h"

class Car {
  public:
    // Class initialization
    Car(const Motor& inEngine, const SteeringWheel& inSteer, const RC& inRemoteController);
    // Press the throttle by a certain strength.
    void accelerate(int strength);

    // Steer the car with a certain speed.
    // Positive and negative values for opposite directions.
    void steer(const int steerSpeed);

    // Steer the car to a position.
    void steerTo(const int steerPos);

    // Brake the car.
    void brake();

    // Run this in main loop to listen to user inputs.
    void listen();

    // Setter of curDriveMode_.
    void setCurDriveMode(uint8_t value);

    void waitForSerial(int timeout=100);

  protected:
    // Listen to Throttle inputs (left vertical).
    void listenThro();
    // Listen to Elev inputs (right vertical)
    void listenElev();
    // Listen to Aile inputs (right horizontal).
    void listenAile();
    // Aux1 inputs are from D toggle with value of 0, 1, or 2.

    // Listen from computer.
    void listenComputer();

    // In Auto drive mode, adjust the steering wheel until it matches
    // targetSteer_.
    void autoSteer();

    // Return current car status.
    String getStatus();
    
  private:
    Motor engine_;
    SteeringWheel steer_;
    RC rc_;
    
    bool engineReverse_ = false;
    int engineMax_ = 255;
    int engineMin_ = 30;

    // Min value of direction A of most RC joysticks.
    const float minA_ = 1102;
    // Throttle has a different low range.
    const float minAThro_ = 1100; 
    // Min value of direction B of most RC joysticks.
    const float maxA_ = 1878;

    // We don't want the car to run forward and backward at the same time
    // so we track both elev and throttle and pick only the one currently active.
    int throSpeed_ = 0;
    int elevSpeed_ = 0;

    bool steerReverse_ = false;
    // Maximum steering power, 0 to 255.
    int steerMax_ = 255;
    // Minimum steering power, 0 to 255.
    int steerMin_ = 20;
    int steerSpeed_ = 0;

    int curSteerFeed_ = 0;
    int steerFeedMin_ = 30;
    int steerFeedMax_ = 993;
    int steerSlack_ = 1;

    uint8_t curDriveMode_ = 0;

    // In Auto mode, if curSteerFeed != targetSteer_,
    // adjust the steering wheel.
    // This is required for `steerTo` method.
    int targetSteer_ = 512;

    // Currently throttle strength, replace when accelerometer installed.
    float curSpeed_ = 0.0;
    
};

#endif
