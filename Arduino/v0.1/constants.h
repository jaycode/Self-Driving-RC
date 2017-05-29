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
// Steering wheel feedback.
const char CMD_STEER = 'S';
// Change drive mode.
const char CMD_CHANGE_DRIVE_MODE = 'D';
//----END ARDUINO COMMANDS----

//----BEGIN COMPUTER COMMANDS----
// Computer asks for wheel feedback.
const char CCMD_STEER = 'S';
// Computer asks for drive mode.
const char CCMD_DRIVE_MODE = 'D';
//----END COMPUTER COMMANDS----

//----BEGIN VALUES----
// Following drive modes are changed by updating AUX1 input.
const uint8_t DRIVE_MODE_MANUAL = 2;
const uint8_t DRIVE_MODE_RECORDED = 1;
const uint8_t DRIVE_MODE_AUTO = 0;
//----END VALUES----

#endif
