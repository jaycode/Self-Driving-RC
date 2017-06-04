# To use, run:
# `sudo su`
# `source activate python3`
# `python main.py --model [model-path]`

# We cannot use any blocking here since some packets do disappear.

import serial
import numpy as np
import cv2
from datetime import datetime
import os
import argparse
import h5py
from keras.models import load_model
import time
import re

DEV_BEGIN = b'B'
DEV_STATUS = b'S'
DEV_DEBUG = b'D'

HOST_REQUEST_UPDATE = b'u'
HOST_AUTO_STEER = b's'
HOST_AUTO_THROTTLE = b't'

MODE_NONE = 0
MODE_LISTEN_STATUS_MODE = 1
MODE_LISTEN_STATUS_SPEED = 2
MODE_LISTEN_STATUS_STEER = 3

# The following values need to be the same with the ones in the microcontroller.
DRIVE_MODE_MANUAL = 1 # 1
DRIVE_MODE_RECORDED = 2 # 2
DRIVE_MODE_AUTO = 0 # 0

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
MAX_THROTTLE = 130
THROTTLE_P=0.3
THROTTLE_I=0.08

def choose_port(ports):
    port_connected = False
    port_idx = 0
    while not port_connected:
        try:
            port = serial.Serial(ports[port_idx], baudrate=9600, timeout=0.05)
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

def read_bytes_until(port, lchar, liter):
    i = 0
    value = ""
    buff = b''
    while buff != lchar and i < liter:
        i+=1
        # If an error happens caused by buff contains \xff or other
        # UnicodeDEcodeError, that means it tries to read more data before
        # it was available. Testing with the following code should help:
        # x = port.read()
        # x1 = port.read()
        # x2 = port.read()
        # x3 = port.read()
        # x4 = port.read()
        # x5 = port.read()
        # print("debug:", (x+x1+x2+x3+x4+x5))
        while port.in_waiting == 0: pass
        value += str(buff, 'latin')
        buff = port.read(1)
    return value

def read_bytes(port, n):
    # This gives better output than port.read(n).
    text = ""
    for i in range(n):
        text += str(port.read(), 'latin')
    port.read() # Final newline
    return text

def read_status(port):
    status = {}
    value = ''
    buff = b''
    while port.in_waiting == 0: pass
    port.read(1)

    # mode
    while buff != b'v':
        while port.in_waiting == 0: pass
        value += str(buff, 'latin')
        buff = port.read(1)
    status['mode'] = int(value)

    value = ''
    buff = b''
    # velocity / speed
    while buff != b's':
        while port.in_waiting == 0: pass
        value += str(buff, 'latin')
        buff = port.read(1)
    status['speed'] = int(value)

    value = ''
    buff = b''
    # steer
    while buff != b'\n':
        while port.in_waiting == 0: pass
        value += str(buff, 'latin')
        buff = port.read(1)
    status['steer'] = int(value)
    return status

def main():
    mode = MODE_NONE
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

    previous_time = time.time()
    while True:
        loop_time = time.time()
        print("single loop time:",loop_time-previous_time)
        previous_time = loop_time

        # --- TESTING STEERING ---
        # port.write(bytearray("{}{};".format(\
        #     HOST_AUTO_STEER.decode(), "0250"), 'utf-8'))

        # while port.in_waiting == 0: pass
        # response = port.read()
        # print("got byte:", time.time()-previous_time)
        # print("response: {}".format(response))
        # if response != b'':
        #     print("connected")
        #     is_connected = True

        # print(read_bytes(port, 6))
        # --- END TESTING STEERING ---


        # if mode == MODE_NONE:
        port.write(HOST_REQUEST_UPDATE)
        # non-blocking check whether device sends BEGIN
        begin = port.read(1)
        if begin == DEV_BEGIN:
            while port.in_waiting == 0: pass
            command = port.read(1)
            if command == DEV_STATUS:
                # blocking read status
                status = read_status(port)
                print(status)
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
                    
                    # Read image and do image preprocessing (when needed)
                    ret, frame = cams[0].read()

                    image_array = np.asarray(frame)
                    throttle = controller.update(status['speed'])
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
                        throttle = int(throttle)
                        new_steer = int(new_steer)
                        print("throttle: {} steer: {}".format(throttle, new_steer))
                        port.write(bytearray("{}{};".format(\
                            HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
                        port.write(bytearray("{}{};".format(\
                            HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))


        # response = b'';
        # if not is_connected:
        #     response = port.read()
        #     print("response: {}".format(response))
        #     if response != b'':
        #         print("connected")
        #         is_connected = True

        # update = str(response, 'latin') + read_bytes(port, 14)
        # if update == '': continue

        # if update[0].encode('latin') == DEV_BEGIN:
        #     command = update[1].encode('latin')

        # print("update: {}".format(update))

        # if command == DEV_DEBUG:
        #     print("debug:", update[2:])
        # elif command == DEV_STATUS:
        #     regex = r'm([0-9])v([-+]?[0-9]*\.?[0-9]+)s([-+]?[0-9]+)'
        #     matches = re.findall(regex, update[2:])[0]
        #     status = {'mode': int(matches[0]),
        #               'speed': float(matches[1]),
        #               'steer': int(matches[2])}
        #     print("status: mode {} speed {} steer {}".format(
        #         status['mode'], status['speed'], status['steer']))

        #     if status['mode'] == DRIVE_MODE_RECORDED:
        #         # Record this frame.
        #         # No need to do image preprocessing here. We want the
        #         # raw image and experiment with preprocessing later in
        #         # training phase. The final preprocessing will then
        #         # be implemented in the inference phase.
        #         ret, frame = cams[0].read()

        #         # Create image path.
        #         filename = "{}.jpg".format(tstamp)
        #         path = os.path.join(RECORDED_IMG_PATH, filename)

        #         # We put the makedirs here to ensure directory is created
        #         # when re-recording without having to reset the script.
        #         os.makedirs(RECORDED_IMG_PATH, exist_ok=True)

        #         # Save image
        #         cv2.imwrite(path, frame)

        #         # Append to training data.
        #         if not os.path.isfile(RECORDED_CSV_PATH):
        #             fd = open(RECORDED_CSV_PATH, 'w')
        #             head = "filename, steer, speed\n"
        #             fd.write(head)
        #         else:
        #             fd = open(RECORDED_CSV_PATH,'a')
        #         row = "{}, {}, {}\n".format(filename, status['steer'], status['speed'])
        #         fd.write(row)
        #         fd.close()
        #     elif status['mode'] == DRIVE_MODE_AUTO:
        #         # Inference phase
                
        #         # Read image and do image preprocessing (when needed)
        #         ret, frame = cams[0].read()

        #         image_array = np.asarray(frame)
        #         throttle = controller.update(status['speed'])
        #         msg = None
        #         try:
        #             prediction = model.predict(\
        #                 image_array[None, :, :, :], batch_size=1)
        #             new_steer = prediction[0][0]
        #         except TypeError as err:
        #             msg = "TypeError: {}".format(err)
        #             print(msg)
        #         except ValueError as err:
        #             msg = "TypeError: {}".format(err)
        #             print(msg)
        #         except UnboundLocalError as err:
        #             # No model since auto mode was disabled.
        #             msg = "AUTO mode was disabled since no model was initialized."
        #             print(msg)
        #         if msg:
        #             with open('error.log','a') as f:
        #                 f.write(msg)
        #                 f.write("\n")
        #         else:
        #             throttle = int(throttle)
        #             new_steer = int(new_steer)
        #             print("throttle: {} steer: {}".format(throttle, new_steer))
        #             port.write(bytearray("{}{};".format(\
        #                 HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
        #             port.write(bytearray("{}{};".format(\
        #                 HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))

if __name__ == "__main__":
    main()

