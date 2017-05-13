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
    Car(const Motor& inEngine, const SteeringWheel& inSteer, const RC& inRemoteController);
    void accelerateTo(int speed);
    void steer(const int steerSpeed);
    void brake();
    uint8_t getSteerSpeed();
    void setSteerSpeed();
    void listen();

  protected:
    void listenThro();
    void listenElev();
    void listenAile();
    void syncSteering(const int steerSpeed);
  private:
    Motor engine_;
    SteeringWheel steer_;
    RC rc_;
    float curAngle_ = 0;
    
    bool engineReverse_ = true;
    int engineMax_ = 255;
    int engineMin_ = 80;

    // We don't want the car to run forward and backward at the same time
    // so we track both elev and throttle and pick only the one currently active.
    int throSpeed_ = 0;
    int elevSpeed_ = 0;

    bool steerReverse_ = true;
    int steerMax_ = 150;
    int steerMin_ = 20;
    float steerAngleMin_ = -60.0;
    float steerAngleMax_ = 60.0;
    float curSteerAngle_ = 0;
    float curRCAngle_ = 0;
    int steerSpeed_ = 0;
    
    // Angle of steer slack.
    float steerSlack_ = 10;
    
    bool braked_ = true;
};

#endif
