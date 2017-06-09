# To use, run:
# `sudo su`
# `python drive.py --model [model-path]

# We cannot use any blocking here since some packets do disappear in the beginning.
# Blocking happens in the microcontroller side.

# Load model routine may generate error due to incompatible python
# compiler used to generate model.h5 file:
# - "Segmentation fault (core dumped)": h5 file created with python 3.6, drive.py uses python 3.5.
# - "SystemError: unknown opcode": h5 file created with python 3.5, drive.py uses python 3.6.

import serial
import cv2
from datetime import datetime
import os
import argparse
import h5py
import time
import re
import glob
import csv
import pickle
from keras.models import load_model
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
CALIBRATION_FILE = os.path.realpath(os.path.join(dir_path, '..', 'calibrations', 'cal-elp.p'))
with open( CALIBRATION_FILE, "rb" ) as pfile:
    cal = pickle.load(pfile)
mtx = cal['mtx']
dist = cal['dist']

DEV_BEGIN = b'B'
DEV_STATUS = b'S'
DEV_DEBUG = b'D'

HOST_REQUEST_UPDATE = b'u'
HOST_AUTO_STEER = b's'
HOST_AUTO_THROTTLE = b't'

# The following values need to be the same with the ones in the microcontroller.
DRIVE_MODE_MANUAL = 1 # 1
DRIVE_MODE_RECORDED = 2 # 2
DRIVE_MODE_AUTO = 0 # 0

RECORDED_IMG_PATH = "/home/sku/recorded"
RECORDED_CSV_PATH = "/home/sku/recorded.csv"
cams = [cv2.VideoCapture(0)]

# Try out several ports to find where the microcontroller is.
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
cams[0].set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
cams[0].set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 45

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

    # Load model routine may generate error due to incompatible python
    # compiler used to generate model.h5 file:
    # - "Segmentation fault (core dumped)": h5 file created with python 3.6, drive.py uses python 3.5.
    # - "SystemError: unknown opcode": h5 file created with python 3.5, drive.py uses python 3.6.
    return load_model(model_path)

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

def preprocess(raw_img):
    # This should be similar to the one in test.py
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)[:, :, 2]
    img = cv2.Sobel(img, -1, 0, 1, ksize=3)
    img = img / 255.0
    img = img > 0.5
    img = np.array([img])
    img = np.rollaxis(np.concatenate((img, img, img)), 0, 3)
    return img[:, :, [0]]

def auto_drive_cams(port, controller, status, model, cams):
    global mtx, dist
    # Read image and do image preprocessing (when needed)
    ret, image = cams[0].read()

    # Preprocessing
    final_img = preprocess(image)
    img_array = np.asarray(final_img)[None, :, :, :]

    throttle = controller.update(status['speed'])
    msg = None
    try:
        prediction = model.predict(\
            img_array, batch_size=1)
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
        print("throttle: {}, steer: {}".format(throttle, new_steer))
        port.write(bytearray("{}{};".format(\
            HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
        port.write(bytearray("{}{};".format(\
            HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))

def main():
    port = choose_port(ports)
    print("Listening for commands...");

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model', type=str,
        help="Path to model definition json. Model weights should be on the same path.")
    parser.add_argument('--recorded-img', type=str,
        default=RECORDED_IMG_PATH,
        help="Path to model definition json. Model weights should be on the same path.")
    parser.add_argument('--recorded-csv', type=str,
        default=RECORDED_CSV_PATH,
        help="Path to model definition json. Model weights should be on the same path.")
    
    args = parser.parse_args()
    if args.model:
        model = prepare_model(args.model)
    else:
        print("Warning: No model has been defined. AUTO mode is disabled.\n"+\
              "Add --model [path to json file] to load a model.")

    previous_time = time.time()
    while True:
        loop_time = time.time()
        # print("single loop time:",loop_time-previous_time)
        previous_time = loop_time
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
                    tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
                    filename = "{}.jpg".format(tstamp)
                    path = os.path.join(args.recorded_img, filename)

                    # We put the makedirs here to ensure directory is created
                    # when re-recording without having to reset the script.
                    os.makedirs(args.recorded_img, exist_ok=True)

                    # Save image
                    cv2.imwrite(path, frame)

                    # Append to training data.
                    if not os.path.isfile(args.recorded_csv):
                        fd = open(args.recorded_csv_path, 'w')
                        head = "filename, steer, speed\n"
                        fd.write(head)
                    else:
                        fd = open(args.recorded_csv,'a')
                    row = "{}, {}, {}\n".format(filename, status['steer'], status['speed'])
                    fd.write(row)
                    fd.close()
                elif status['mode'] == DRIVE_MODE_AUTO:
                    # Inference phase                        
                    auto_drive_cams(port, controller, status, model, cams)

if __name__ == "__main__":
    main()
