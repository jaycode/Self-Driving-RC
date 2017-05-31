import serial
import numpy as np
import cv2
from datetime import datetime
import os
import argparse
import h5py
from keras.models import load_model


CMD_BEGIN = b'C'
CMD_STEER = b'S'
CMD_CHANGE_DRIVE_MODE = b'D'
CMD_AUTO = b'A'
CMD_DEBUG = b'd'
# `iv200;o512;` for velocity 200 and orientation 512.
CMD_REQUEST_INSTRUCTIONS = b'i';


CCMD_REQUEST_STEER = b'S'
CCMD_DRIVE_MODE = b'D'
CCMD_AUTO_STEER = b's'
CCMD_AUTO_THROTTLE = b't'

# These modes are just used internally by the computer to decide
# what mode it is currently in.
# For example, mode NONE means the system is at standby.
# When the microcontroller sends a command CMD_BEGIN (or, simply put,
# `Serial.print("C")` was initiated by Arduino), computer changes its mode to
# MODE_LISTEN_CMD where it prepares to read the next command.
#
# Some commands may be followed by steering value, let's look at one example:
# First, the computer asks for steering value, it does so by sending a
# character 'S' (CCMD_REQUEST_STEER).
# When the microcontroller reads this, it sends back a char 'C', followed
# by 'S' (CMD_STEER). Computer then switches to MODE_LISTEN_STEER_VAL.
# And finally, the microcontroller sends the number of steering value.
# accepted by the computer.
MODE_NONE = 0
MODE_LISTEN_CMD = 1
MODE_LISTEN_STEER_VAL = 2
MODE_LISTEN_DRIVE_MODE = 3
MODE_LISTEN_DEBUG = 4
MODE_LISTEN_STATUS = 5

# The following values need to be the same with the ones in the microcontroller.
DRIVE_MODE_MANUAL = 1
DRIVE_MODE_RECORDED = 2
DRIVE_MODE_AUTO = 0

RECORDED_IMG_PATH = "/home/sku/recorded"
RECORDED_CSV_PATH = "/home/sku/recorded.csv"
cams = [cv2.VideoCapture(0)]
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 230

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

class SimplePIController:
    def __init__(self, Kp, Ki):
        self.Kp = Kp
        self.Ki = Ki
        self.set_point = 0.
        self.error = 0.
        self.integral = 0.

    def set_desired(self, desired):
        self.set_point = desired

    def update(self, measurement):
        # proportional error
        self.error = self.set_point - measurement

        # integral error
        self.integral += self.error

        return self.Kp * self.error + self.Ki * self.integral


controller = SimplePIController(0.1, 0.002)
controller.set_desired(set_speed)

def prepare_model(model_path):
    from keras import __version__ as keras_version
    # check that model Keras version is same as local Keras version
    f = h5py.File(model_path, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    return load_model(model_path)

def read_bytes_until(port, lchar, liter):
    i = 0
    value = ""
    buff = b''
    while buff != lchar and i < liter:
        i+=1
        value += str(buff, 'utf-8')
        buff = port.read(1)
    return value

def main():
    port = choose_port(ports)
    mode = MODE_NONE

    # Default mode cannot be set since when the remote controller
    # is off it directly defaults to MANUAL.
    # One way to set to other mode without turning on the remote
    # controller is by switch the value of another drive mode with
    # DRIVE_MODE_MANUAL and do the same thing in the microcontroller side.
    drive_mode = None

    # Asking for drive mode to microcontroller.
    port.write(CCMD_DRIVE_MODE)

    cur_steer = 0;

    buff = ''
    print("Listening for commands...");
    cycle = 0


    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    model = prepare_model(args.model)

    while True:
        cycle+=1
        if cycle == 100000:
            cycle = 0;
        if drive_mode == DRIVE_MODE_RECORDED:
            if cycle%1 == 0:
                # Computer asks for steering wheel value on RECORDED mode,
                # and it does so every cycle.
                port.write(CCMD_REQUEST_STEER)
        elif drive_mode == DRIVE_MODE_AUTO:
            if MODE_LISTEN_STATUS:
                s_cmd = ''
                velocity = 0.0
                orientation = 512
                while s_cmd != b"\n":
                    s_cmd = port.read(1)
                    if s_cmd == b'v':
                        velocity = float(read_bytes_until(port, b';', 8))
                    elif s_cmd == b'o':
                        orientation = int(read_bytes_until(port, b';', 4))

                print("Car status: v: {}o: {}".format(velocity, orientation))

                ret, frame = cams[0].read()
                image_array = np.asarray(frame)
                throttle = controller.update(velocity)
                steering = model.predict(image_array[None, :, :, :], batch_size=1)

                port.write("{}{};".format(CCMD_AUTO_STEER, str(steering)))
                port.write("{}{};".format(CCMD_AUTO_THROTTLE, str(throttle)))

        if mode == MODE_NONE:
            cmd = port.read(1)
            if cmd == CMD_BEGIN:
                mode = MODE_LISTEN_CMD
        elif mode == MODE_LISTEN_CMD:
            cmd = port.read(1)
            # Add here whenever a new command is added.
            if cmd == CMD_STEER:
                mode = MODE_LISTEN_STEER_VAL
            elif cmd == CMD_CHANGE_DRIVE_MODE:
                mode = MODE_LISTEN_DRIVE_MODE
            elif cmd == CMD_DEBUG:
                mode = MODE_LISTEN_DEBUG
            elif cmd == CMD_REQUEST_INSTRUCTIONS:
                mode = MODE_LISTEN_STATUS
        elif mode == MODE_LISTEN_STEER_VAL:
            val = port.read(1)
            if val != b"\n":
                buff += str(val, 'utf-8')
            else:
                cur_steer = int(buff)

                # Get timestamp and steer information.
                tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
                print("{} steer: {}".format(tstamp, cur_steer))
                
                if drive_mode == DRIVE_MODE_RECORDED:
                    # Record this frame.
                    ret, frame = cams[0].read()

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
            if val == DRIVE_MODE_RECORDED:
                # Recorded mode
                drive_mode = DRIVE_MODE_RECORDED
                print("Set drive mode to RECORDED")
            elif val == DRIVE_MODE_AUTO:
                drive_mode = DRIVE_MODE_AUTO
                print("Set drive mode to AUTO")
            elif val == DRIVE_MODE_MANUAL:
                # Manual mode
                drive_mode = DRIVE_MODE_MANUAL
                print("Set drive mode to MANUAL")
            mode = MODE_NONE
        elif mode == MODE_LISTEN_DEBUG:
            val = port.read(1)
            if val != b"\n":
                buff += str(val, 'utf-8')
            else:
                print(buff);

                # Reset buffer
                buff = ''

                # Reset mode back to NONE
                mode = MODE_NONE


if __name__ == "__main__":
    main()

