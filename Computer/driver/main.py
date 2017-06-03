# To use, run:
# `sudo su`
# `source activate python3`
# `python main.py --model [model-path]`
import serial
import numpy as np
import cv2
from datetime import datetime
import os
import argparse
import h5py
from keras.models import load_model
import time

CMD_BEGIN = b'C'
CMD_STATUS = b'S'
CMD_CHANGE_DRIVE_MODE = b'D'
CMD_AUTO = b'A'
CMD_DEBUG = b'd'
# `iv200;o512;` for velocity 200 and orientation 512.
CMD_REQUEST_INSTRUCTIONS = b'i';


CCMD_REQUEST_STATUS = b'S'
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
# by 'S' (CMD_STATUS). Computer then switches to MODE_LISTEN_STATUS.
# And finally, the microcontroller sends the number of steering value.
# accepted by the computer.
MODE_NONE = 0
MODE_LISTEN_CMD = 1
MODE_LISTEN_STATUS = 2
MODE_LISTEN_DRIVE_MODE = 3
MODE_LISTEN_DEBUG = 4

# The following values need to be the same with the ones in the microcontroller.
DRIVE_MODE_MANUAL = 0
DRIVE_MODE_RECORDED = 2
DRIVE_MODE_AUTO = 1

RECORDED_IMG_PATH = "/home/sku/recorded"
RECORDED_CSV_PATH = "/home/sku/recorded.csv"
cams = [cv2.VideoCapture(0)]
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
cams[0].set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cams[0].set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)
TARGET_CROP = ((70, 20), (0, 0))

# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 130

MIN_THROTTLE = 50
MAX_THROTTLE = 255
THROTTLE_P=0.25
THROTTLE_I=0.01

def choose_port(ports):
    port_connected = False
    port_idx = 0
    while not port_connected:
        try:
            port = serial.Serial(ports[port_idx], baudrate=9600, timeout=None)
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


controller = SimplePIController(THROTTLE_P, THROTTLE_I)
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

def listen_status(port):
    s_cmd = ''
    speed = 0.0
    steer = 512
    while s_cmd != b"\n":
        time_a = time.time()
        s_cmd = port.read(1)
        print("status a:", time.time() - time_a)
        print(s_cmd)
        if s_cmd == b'v':
            time_b = time.time()
            speed = float(read_bytes_until(port, b';', 8))
            print("status b:", time.time() - time_b)
        elif s_cmd == b'o':
            time_c = time.time()
            steer = int(read_bytes_until(port, b';', 5))
            print("status c:", time.time() - time_c)

    # There will be some buffer leftover when user changes the drive mode,
    # that is why this assert is commented out. We keep it here for
    # debugging.
    # waiting = port.in_waiting
    # assert (waiting == 0), "Buffer leftover in listen_status: {}".format(port.read(waiting))
    return (steer, speed)

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
    parser.add_argument('--model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    if args.model:
        model = prepare_model(args.model)
    else:
        print("Warning: No model has been defined. AUTO mode is disabled.\n"+\
              "Add --model [path to json file] to load a model.")

    previous_time = time.time()
    while True:
        loop_time = time.time()
        print("single loop time:",loop_time-previous_time)
        prevous_time = loop_time

        cycle+=1
        if cycle == 100000:
            cycle = 0

        if mode == MODE_NONE:
            # All communication should be initiated by the computer, since
            # otherwise there will be some data left in the buffer that would
            # cause in lags. That is why we send request to gain information
            # here.
            if drive_mode == DRIVE_MODE_RECORDED or drive_mode == DRIVE_MODE_AUTO:
                port.write(CCMD_REQUEST_STATUS)
                mode = MODE_LISTEN_STATUS
                
            cmd = port.read(1)
            if cmd == CMD_BEGIN:
                mode = MODE_LISTEN_CMD
        elif mode == MODE_LISTEN_CMD:
            cmd = port.read(1)
            # Add here whenever a new command is added.
            if cmd == CMD_STATUS or cmd == CMD_REQUEST_INSTRUCTIONS:
                mode = MODE_LISTEN_STATUS
            elif cmd == CMD_CHANGE_DRIVE_MODE:
                mode = MODE_LISTEN_DRIVE_MODE
            elif cmd == CMD_DEBUG:
                mode = MODE_LISTEN_DEBUG


        if mode == MODE_LISTEN_STATUS:
            # Get timestamp and steer information.
            steer, speed = listen_status(port)
            tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
            print("steer: {} speed: {} ({})".format(steer, speed, tstamp))

            print("mid time:", time.time() - loop_time)
            if drive_mode == DRIVE_MODE_RECORDED:
                # Record this frame.
                # No need to do image preprocessing here. We want the
                # raw image and experiment with preprocessing later in
                # training phase. The final preprocessing will then
                # be implemented in the inference phase.
                ret, frame = cams[0].read()

                # Create image path.
                filename = "{}.jpg".format(tstamp)
                path = os.path.join(RECORDED_IMG_PATH, filename)

                # We put the makedirs here to ensure directory is created
                # when re-recording without having to reset the script.
                os.makedirs(RECORDED_IMG_PATH, exist_ok=True)

                # Save image
                cv2.imwrite(path, frame)

                # Append to training data.
                if not os.path.isfile(RECORDED_CSV_PATH):
                    fd = open(RECORDED_CSV_PATH, 'w')
                    head = "filename, steer, speed\n"
                    fd.write(head)
                else:
                    fd = open(RECORDED_CSV_PATH,'a')
                row = "{}, {}, {}\n".format(filename, steer, speed)
                fd.write(row)
                fd.close()

            elif drive_mode == DRIVE_MODE_AUTO:
                # Inference phase

                start = time.time()
                
                # Read image and do image preprocessing
                ret, frame = cams[0].read()

                image_array = np.asarray(frame)
                throttle = controller.update(speed)
                msg = None
                try:
                    prediction = model.predict(\
                        image_array[None, :, :, :], batch_size=1)
                    new_steer = prediction[0][0]
                except TypeError as err:
                    msg = "TypeError: {}".format(err)
                    print(msg)
                except ValueError as err:
                    msg = "TypeError: {}".format(err)
                    print(msg)
                except UnboundLocalError as err:
                    # No model since auto mode was disabled.
                    msg = "AUTO mode was disabled since no model was initialized."
                    print(msg)
                if msg:
                    with open('error.log','a') as f:
                        f.write(msg)
                        f.write("\n")
                else:
                    print("throttle:", throttle)
                    port.write(bytearray("{}{};".format(\
                        CCMD_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))
                    port.write(bytearray("{}{};".format(\
                        CCMD_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
                end = time.time()
                print("time:", end-start)
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
                port.write(bytearray("{}{};".format(\
                    CCMD_AUTO_THROTTLE.decode(), str(MIN_THROTTLE)), 'utf-8'))
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

