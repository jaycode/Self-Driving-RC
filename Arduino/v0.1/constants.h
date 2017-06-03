/*
 * Constants. Make sure the computer part contains all of them. 
 */
// Include guard
// See: http://en.wikipedia.org/wiki/Include_guard
#ifndef _CONSTANTS_H_
#define _CONSTANTS_H_

#include <stdint.h>

//----BEGIN ARDUINO COMMANDS----
// Begin listening to the next command.
const char CMD_BEGIN = 'C';
// Steering wheel and speed feedback.
const char CMD_STATUS = 'S';
// Change drive mode.
const char CMD_CHANGE_DRIVE_MODE = 'D';
// Debug. Send message to computer to display.
const char CMD_DEBUG = 'd';
// Request instructions. Pass this along with vehicle info
// like velocity and orientation. Example command:
// `iv200;o512;` for velocity 200 and orientation 512.
const char CMD_REQUEST_INSTRUCTIONS = 'i';
//----END ARDUINO COMMANDS----

//----BEGIN COMPUTER COMMANDS----
// Computer asks for wheel feedback.
const char CCMD_REQUEST_STATUS = 'S';
// Computer asks for drive mode.
const char CCMD_DRIVE_MODE = 'D';
// Computer steers the wheel.
const char CCMD_AUTO_STEER = 's';
// Computer inputs throttle.
const char CCMD_AUTO_THROTTLE = 't';
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
const uint8_t DRIVE_MODE_MANUAL = 0; // 1
const uint8_t DRIVE_MODE_RECORDED = 2;
const uint8_t DRIVE_MODE_AUTO = 1;
//----END VALUES----

#endif
