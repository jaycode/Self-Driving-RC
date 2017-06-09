# To use, run:
# `sudo su`
# `python drive.py [--model] [-t] [-v]

# === Model Path ===
# `model` is the path to model definition h5. Model definition is created by learner/learn.py script.

# === Throttle ===
# By default, this script does not actuate throttle. To allow it to
# send throttle commands, include flag `-t`.

# === Visualize ===
# `-v` option visualizes the what the car sees.

# === Socket ===
# We cannot use any blocking here since some packets do disappear in the beginning.
# Blocking happens in the microcontroller side.

# === WARNING ===
# Load model routine may generate error due to incompatible python
# compiler used to generate model.h5 file:
# - "Segmentation fault (core dumped)": h5 file created with python 3.6, drive.py uses python 3.5.
# - "SystemError: unknown opcode": h5 file created with python 3.5, drive.py uses python 3.6.

import numpy as np
import sys
import cv2
from datetime import datetime
import os
import argparse
import time
import re
import glob
import csv
import pickle
from keras.models import load_model
from threading import Thread
from queue import Queue
import pdb

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, choose_port, preprocess, prepare_model

# Path to calibration file
CALIBRATION_FILE = os.path.realpath(os.path.join(dir_path, '..', 'calibrations', 'cal-elp.p'))
with open( CALIBRATION_FILE, "rb" ) as pfile:
    cal = pickle.load(pfile)
mtx = cal['mtx']
dist = cal['dist']

config = configuration()

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

# Try out several ports to find where the microcontroller is.
ports = ["/dev/ttyUSB0", "/dev/ttyUSB1"]

# This is the smallest current camera may support.
# (i.e. setting CAP_PROP_FRAME_WIDTH and HEIGHT smaller than this won't help)
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']

# TODO: This is currently throttle value but we will update it once we got
#       accelerometer.
set_speed = 45

MIN_THROTTLE = 50
MAX_THROTTLE = 130
THROTTLE_P=0.3
THROTTLE_I=0.08

# Record latency in seconds.
# This variable stores the time between each recording session. 
# The larger this value, the more we accommodate recording time, which means the latency between
# status and image collection (`rec_latency` field in `recorded.csv`) gets lower, but there will
# be more time between each image recording.
# Hence, good value needs to match the `rec_latency` field in `recorded.csv`, but not much more.
# TODO: If less latency is needed, store the images and status in memory and then store them in batches.
REC_LATENCY_SEC=0.3

# === RECORDER ===
# Record and process images in a separate thread
img_queue = Queue(maxsize=128)

def draw_visualization(final_img, image, steer=None):
    f3 = np.stack((final_img[:, :, 0], final_img[:, :, 0], final_img[:, :, 0]), axis=2)
    f3 = f3[TARGET_CROP[0][0]:(TARGET_HEIGHT-TARGET_CROP[0][1]),
            TARGET_CROP[1][0]:(TARGET_WIDTH-TARGET_CROP[1][1]), :]
    f3 = (f3 * 255.0).astype(np.uint8)

    text1 = "steer: {}".format(steer)
    f3 = cv2.putText(f3, text1, (10,50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 210))

    viz = np.concatenate((f3, image), axis=0)
    cv2.imshow("RC", viz)
    cv2.waitKey(1)

def record(cams):
    while True:
        if img_queue.qsize() > 0:
            item = img_queue.get()

            # # We put the makedirs here to ensure directory is created
            # # when re-recording without having to reset the script.
            os.makedirs(item['img_dir_path'], exist_ok=True)

            # Record this frame.
            # No need to do image preprocessing here. We want the
            # raw image and experiment with preprocessing later in
            # training phase. The final preprocessing will then
            # be implemented in the inference phase.
            ret, frame = cams[0].read()

            cv2.imwrite(item['img_path'], frame)
            # Append to training data.
            if not os.path.isfile(item['csv_path']):
                fd = open(item['csv_path'], 'w')
                head = "filename, steer, speed, rec_latency\n"
                fd.write(head)
            else:
                fd = open(item['csv_path'],'a')
            row = "{}, {}, {}, {}\n".format(item['img_name'], item['status']['steer'], item['status']['speed'], (time.time() - item['time']))
            fd.write(row)
            fd.close()
            img_queue.task_done()

# === END RECORDER ===

def find_cams(num=1, n_ports=4):
    """ Find active cameras
    Args:
    - num: Number of cameras to find.
    - n_ports: Maximum number of ports to try.
    """
    cams = []
    for i in range(n_ports):
        cam = cv2.VideoCapture(i)
        ret, img = cam.read()
        if ret:
            print("Camera {} online".format(i))
            cams.append(cam)
        if len(cams) == num:
            break
    return cams


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

def auto_drive_cams(port, controller, status, model, cams, allow_throttle, visualize=False):
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
        if allow_throttle:
            port.write(bytearray("{}{};".format(\
                HOST_AUTO_THROTTLE.decode(), str(throttle)), 'utf-8'))
        port.write(bytearray("{}{};".format(\
            HOST_AUTO_STEER.decode(), str(new_steer)), 'utf-8'))

        # Visualize

        if visualize:
            draw_visualization(final_image, image, steer=new_steer)

def main():
    port = choose_port(ports)
    print("Listening for commands...");

    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('--model', type=str,
        help="Path to model definition h5 file. Model definition is created by learner/learn.py script.")
    parser.add_argument('--recorded-img', type=str,
        default=RECORDED_IMG_PATH,
        help="Path to recorded image's directory.")
    parser.add_argument('--recorded-csv', type=str,
        default=RECORDED_CSV_PATH,
        help="Path to recorded csv file.")
    parser.add_argument('-t', action='store_true',
        default=False,
        help="By default, this script does not actuate throttle. To allow it to"
        "send throttle commands, include flag `-t`.")
    parser.add_argument('-v', action='store_true',
        default=False,
        help="`-v` option visualizes the what the car sees.")

    args = parser.parse_args()
    if args.model:
        model = prepare_model(args.model)
    else:
        print("Warning: No model has been defined. AUTO mode is disabled.\n"+\
              "Add --model [path to json file] to load a model.")

    allow_throttle = args.t
    visualize = args.v

    cams = find_cams(num=1, n_ports=4)
    for cam in cams:
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, TARGET_WIDTH)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, TARGET_HEIGHT)

    if len(cams) == 0:
        raise Exception("Camera is not properly connected. It usually helps to re-plug its USB cable.")

    img_thread = Thread(target=record, args=[cams])
    img_thread.start()

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
                    # Create image path.
                    tstamp = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S.%f")
                    filename = "{}.jpg".format(tstamp)
                    path = os.path.join(args.recorded_img, filename)

                    # Save image

                    img_queue.put({
                        'csv_path': args.recorded_csv,
                        'img_dir_path': args.recorded_img,
                        'img_name': filename,
                        'img_path': path,
                        'status': status,
                        'time': time.time()
                    })

                    time.sleep(REC_LATENCY_SEC)
                    if visualize:
                        ret, image = cams[0].read()

                        # Preprocessing
                        final_img = preprocess(image)
                        draw_visualization(final_img, image, steer=status['steer'])

                elif status['mode'] == DRIVE_MODE_AUTO:
                    # Inference phase
                    auto_time_1 = time.time()
                    auto_drive_cams(port, controller, status, model, cams, allow_throttle, visualize)
                    print("auto latency:", (time.time() - auto_time_1))

                elif status['mode'] == DRIVE_MODE_MANUAL:
                    if visualize:
                        ret, image = cams[0].read()

                        # Preprocessing
                        final_img = preprocess(image)
                        draw_visualization(final_img, image, steer=status['steer'])

if __name__ == "__main__":
    main()
