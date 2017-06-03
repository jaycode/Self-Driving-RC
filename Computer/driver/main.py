import serial
import numpy as np
import cv2
from datetime import datetime
import os
import argparse
import h5py
from keras.models import load_model
import re


DEV_BEGIN = b'B'
DEV_STATUS = b'S'
DEV_DEBUG = b'D'


HOST_REQUEST_UPDATE = b'u'
HOST_AUTO_STEER = b's'
HOST_AUTO_THROTTLE = b't'

# The following values need to be the same with the ones in the microcontroller.
DRIVE_MODE_MANUAL = 1
DRIVE_MODE_RECORDED = 2
DRIVE_MODE_AUTO = 0

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
set_speed = 100

MIN_THROTTLE = 50

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

# def listen_status(port):
#     s_cmd = ''
#     speed = 0.0
#     steer = 512
#     while s_cmd != b"\n":
#         s_cmd = port.read(1)
#         if s_cmd == b'v':
#             speed = float(read_bytes_until(port, b';', 8))
#         elif s_cmd == b'o':
#             steer = int(read_bytes_until(port, b';', 5))

#     # There will be some buffer leftover when user changes the drive mode,
#     # that is why this assert is commented out. We keep it here for
#     # debugging.
#     # waiting = port.in_waiting
#     # assert (waiting == 0), "Buffer leftover in listen_status: {}".format(port.read(waiting))
#     return (steer, speed)

def read_bytes_until(port, lchar, liter):
    i = 0
    value = ""
    buff = b''
    while buff != lchar and i < liter:
        i+=1
        value += str(buff, 'utf-8')
        buff = port.read(1)
    return value

def request_device_update(port):
    status = {}
    update = read_bytes_until(port, b"\n", 100)
    if update[0] == DEV_BEGIN:
        command = update[1]
        regex = ur"\m([0-9])\v([-+]?[0-9]+)\;\s([-+]?[0-9]*\.?[0-9]+)"
        matches = re.findall(regex, update[2:])
        status['mode'] = matches[0]
        status['speed'] = matches[1]
        status['steer'] = matches[2]
    return 

def main():
    port = choose_port(ports)

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

    while True:
        cycle+=1
        if cycle == 100000:
            cycle = 0

        status, command = request_device_update(port)
        if command == DEV_DEBUG:
            print
        elif command == DEV_STATUS:
            if status['mode'] == DRIVE_MODE_RECORDED:
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
                row = "{}, {}, {}\n".format(filename, status['steer'], status['speed'])
                fd.write(row)
                fd.close()
            elif status['mode'] == DRIVE_MODE_AUTO:
                # Inference phase
                
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
                        HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))
                    port.write(bytearray("{}{};".format(\
                        HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))

# ===================================

        if mode == MODE_NONE:
            if port.in_waiting > 0:
                cmd = port.read(1)
                if cmd == DEV_BEGIN:
                    mode = MODE_LISTEN_DEV
            else:
                # All communication should be initiated by the computer, since
                # otherwise there will be some data left in the buffer that would
                # cause in lags. That is why we send request to gain information
                # here.
                if drive_mode == DRIVE_MODE_RECORDED or drive_mode == DRIVE_MODE_AUTO:
                    port.write(HOST_REQUEST_STATUS)
                    mode = MODE_LISTEN_STATUS
        elif mode == MODE_LISTEN_DEV:
            cmd = port.read(1)
            # Add here whenever a new command is added.
            if cmd == DEV_STATUS or cmd == DEV_REQUEST_INSTRUCTIONS:
                mode = MODE_LISTEN_STATUS
            elif cmd == DEV_CHANGE_DRIVE_MODE:
                mode = MODE_LISTEN_DRIVE_MODE
            elif cmd == DEV_DEBUG:
                mode = MODE_LISTEN_DEBUG
        if mode == MODE_LISTEN_STATUS:
            # Get timestamp and steer information.
            steer, speed = listen_status(port)
            tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
            print("steer: {} speed: {} ({})".format(steer, speed, tstamp))

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
                
                # Read image and do image preprocessing
                ret, frame = cams[0].read()

                # Crop frame and use certain layer(s). See `learner/learner.py` in
                # both `generate()` function and `input_shape` parameter of the model.
                frame = frame[TARGET_CROP[0][0]:(TARGET_HEIGHT - TARGET_CROP[0][1]),
                              TARGET_CROP[1][0]:(TARGET_HEIGHT - TARGET_CROP[1][1]), :]

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
                        HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))
                    port.write(bytearray("{}{};".format(\
                        HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
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

