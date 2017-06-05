/*
 * Constants. Make sure the computer part contains all of them. 
 */
// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <stdint.h>

//----BEGIN ARDUINO COMMANDS----
// Tells the host to begin listening to the next command.
const char DEV_BEGIN = 'B';
// Steering wheel and speed feedback.
const char DEV_STATUS = 'S';
// Debug. Send message to computer to display.
const char DEV_DEBUG = 'D';
//----END ARDUINO COMMANDS----

//----BEGIN COMPUTER COMMANDS----
// Computer asks for wheel feedback.
const char HOST_REQUEST_UPDATE = 'u';
// Computer steers the wheel.
const char HOST_AUTO_STEER = 's';
// Computer inputs throttle.
const char HOST_AUTO_THROTTLE = 't';
//----END COMPUTER COMMANDS----

//----BEGIN VALUES----
// Following drive modes are changed by updating AUX1 input.
// Interestingly, when DX6 RC turned off, it automatically switch
// to position 1 (i.e. center position), this is why we set position 1
// as MANUAL MODE (we don't want the car to start recording when
// remote controller is turned off).
// 
// WARNING: Turning the remote controller off will make the feed slower
//          (depending on the `timeout` setting in RC class).
const uint8_t DRIVE_MODE_MANUAL = 1; // 1
const uint8_t DRIVE_MODE_RECORDED = 2; // 2
const uint8_t DRIVE_MODE_AUTO = 0; // 0
//----END VALUES----

#endif
