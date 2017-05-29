import serial
import numpy as np
import cv2
from datetime import datetime
import os

CMD_BEGIN = 'C'
CMD_STEER = 'S'
CMD_CHANGE_DRIVE_MODE = 'D'

CCMD_STEER = 'S'
CCMD_DRIVE_MODE = 'D'

MODE_NONE = 0
MODE_LISTEN_CMD = 1
MODE_LISTEN_STEER_VAL = 2
MODE_LISTEN_DRIVE_MODE = 3

DRIVE_MODE_MANUAL = 2
DRIVE_MODE_RECORDED = 1
DRIVE_MODE_AUTO = 0

RECORDED_IMG_PATH = "/home/sku/recorded"
RECORDED_CSV_PATH = "/home/sku/recorded.csv"
cams = [cv2.VideoCapture(0)]
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

def choose_port(ports):
    port_connected = False
    port_idx = 0
    while not port_connected:
        try:
            port = serial.Serial(ports[port_idx], baudrate=9600, timeout=0.1)
            port_connected = True
            print("Port {} connected!\n".format(ports[port_idx]))
        except:
            print("Port {} not connected".format(ports[port_idx]))
            port_idx+=1;
            if len(ports) > port_idx:
                print(", trying {}...".format(ports[port_idx]))
            else:
                print("No other port to try.")
                exit();
    return port


def main():
    port = choose_port(ports)
    mode = MODE_NONE

    drive_mode = DRIVE_MODE_MANUAL

    # Asking for drive mode to Arduino.
    port.write(CCMD_DRIVE_MODE)

    cur_steer = 0;

    buff = ''
    print("Listening for commands...");
    cycle = 0
    while True:
        cycle+=1
        if drive_mode == DRIVE_MODE_RECORDED:
            if cycle%1 == 0:
                port.write(CCMD_STEER)

        if mode == MODE_NONE:
            cmd = port.read(1)
            if cmd == CMD_BEGIN:
                mode = MODE_LISTEN_CMD
        elif mode == MODE_LISTEN_CMD:
            cmd = port.read(1)
            if cmd == CMD_STEER:
                mode = MODE_LISTEN_STEER_VAL
            elif cmd == CMD_CHANGE_DRIVE_MODE:
                mode = MODE_LISTEN_DRIVE_MODE
        elif mode == MODE_LISTEN_STEER_VAL:
            val = port.read(1)
            if val != "\n":
                buff += val
            else:
                cur_steer = int(buff)

                # Record this frame.
                ret, frame = cams[0].read()

                # Get timestamp and steer information.
                tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
                print("{} steer: {}".format(tstamp, cur_steer))
                
                # Create image path.
                filename = "{}.jpg".format(tstamp)
                path = os.path.join(RECORDED_IMG_PATH, filename)
                
                # Save image
                cv2.imwrite(path,frame)

                # Append to training data.
                if not os.path.isfile(RECORDED_CSV_PATH):
                    fd = open(RECORDED_CSV_PATH, 'w')
                    head = "steer, speed, filename\n"
                    fd.write(head)
                else:
                    fd = open(RECORDED_CSV_PATH,'a')
                row = "{}, {}, {}\n".format(cur_steer, 0, filename)
                fd.write(row)
                fd.close()

                # Reset buffer
                buff = ''

                # Reset mode back to NONE
                mode = MODE_NONE
        elif mode == MODE_LISTEN_DRIVE_MODE:
            val = int(port.read(1))
            if val == DRIVE_MODE_MANUAL:
                # Manual mode
                drive_mode = DRIVE_MODE_MANUAL
                print("Set drive mode to MANUAL")
            elif val == DRIVE_MODE_RECORDED:
                # Recorded mode
                drive_mode = DRIVE_MODE_RECORDED
                print("Set drive mode to RECORDED")
            elif val == DRIVE_MODE_AUTO:
                drive_mode = DRIVE_MODE_AUTO
                print("Set drive mode to AUTO")
            mode = MODE_NONE

if __name__ == "__main__":
    main()

