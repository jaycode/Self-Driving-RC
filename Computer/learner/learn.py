import tensorflow as tf
import csv
import cv2
import os
import glob
import argparse
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import pickle

import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping1D, Cropping2D
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam

# Look into Arduino code's car.h for SteerFeedMin_ and SteerFeedMax_
STEER_MIN = 30
STEER_MAX = 993

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = 320
TARGET_HEIGHT = 240
TARGET_CROP = ((60, 20), (0, 0))

BATCH_SIZE=32
EPOCHS=5

DATA_DIRS = [
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1\\",
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-04\\"
"/home/jay/Self-Driving-RC-Data/recorded-2017-06-01.01/",
"/home/jay/Self-Driving-RC-Data/recorded-2017-06-04/"
]

dir_path = os.path.dirname(os.path.realpath(__file__))
CALIBRATION_FILE = os.path.realpath(os.path.join(dir_path, '..', 'calibrations', 'cal-elp.p'))

# ID of center image filename in the csv file.
FILENAME_CENTER_FIELD_ID = 0
STEER_FIELD_ID = 1
SPEED_FIELD_ID = 2

# The model is going to be created at this path. If the model
# already exists at that location, use it as the base for subsequent learning.
# MODEL_H5 = os.path.abspath('C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\model.h5')
MODEL_H5 = os.path.abspath('/home/jay/Self-Driving-RC-Data/model.h5')

with open( CALIBRATION_FILE, "rb" ) as pfile:
    cal = pickle.load(pfile)
mtx = cal['mtx']
dist = cal['dist']

parser = argparse.ArgumentParser(description='Preprocess images')
parser.add_argument('model', type=str,
    default=MODEL_H5,
    help="Path to output model H5 file e.g. `/home/user/model.h5`.")
parser.add_argument('--data-dirs', type=str,
    default=DATA_DIRS,
    nargs='+',
    help="Path to directory that contains csv file and a directory of images.")
args = parser.parse_args()
data_dirs = args.data_dirs
model_h5 = args.model

lines = []
csv_files = data_dirs
for i, data_dir in enumerate(data_dirs):
    # Get the first csv file in the dir.
    data_file = glob.glob(os.path.join(data_dir, '*.csv'))[0]
    with open(data_file) as csvfile:
        reader = csv.reader(csvfile)
        next(reader, None)
        for line in reader:
            # Convert filename to complete path
            imgdir = os.path.abspath(os.path.join(data_dir, 'recorded'))
            imgpath = os.path.join(
                imgdir,
                os.path.basename(line[FILENAME_CENTER_FIELD_ID]))
            line[FILENAME_CENTER_FIELD_ID] = imgpath

            line.append(i)
            lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

def preprocess(raw_img):
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)[:, :, 2]
    img = cv2.Sobel(img, -1, 0, 1, ksize=3)
    img = img / 255.0
    img = img > 0.5
    return img

def generator(samples, batch_size=32):
    steer_range = STEER_MAX - STEER_MIN
    steer_mid = (steer_range/2) + STEER_MIN

    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # TODO: Use multiple cameras
                # corrections = [0, (0.2*steer_range), (0.2*steer_range)] # center, left, right
                corrections = [0]

                # The following loop takes data from three cameras: center, left, and right.
                # The steering measurement for each camera is then added by
                # the correction as listed above.
                for i, c in enumerate(corrections):
                    # field number i contains the image.
                    source_path = batch_sample[i]
                    print("path:", source_path)
                    image = cv2.imread(source_path)

                    # Preprocessing
                    image = preprocess(image)

                    images.append(image)

                    steer_from_mid = float(batch_sample[STEER_FIELD_ID])-steer_mid
                    measurement = steer_mid + steer_from_mid + c
                    measurements.append(int(measurement))

                    # Flip
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurement_flipped = steer_mid - steer_from_mid - c
                    measurements.append(int(measurement_flipped))

            X_train = np.array(images).astype('float')
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

if os.path.exists(MODEL_H5):
    model = load_model(MODEL_H5)
else:
    # Model building
    model = Sequential()
    model.add(Lambda(lambda x: x/255.0,
        input_shape=(TARGET_HEIGHT,TARGET_WIDTH, 3)))
    # YUV Normalization
    # model.add(Lambda(
    # lambda x: (x - 16) / (np.matrix([235.0, 240.0, 240.0]) - 16) - 0.5,
    # input_shape=(TARGET_HEIGHT,TARGET_WIDTH, 3)))

    model.add(Cropping2D(cropping=TARGET_CROP))
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    
    # P3 Jay
    # model.add(Cropping2D(cropping=TARGET_CROP))
    # model.add(Dropout(0.1))
    # model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    # model.add(Dropout(0.25))
    # model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Conv2D(64, (3, 3), activation='relu'))
    # model.add(Dropout(0.5))
    # model.add(Flatten())
    # model.add(Dense(100))
    # model.add(Dense(50))
    # model.add(Dense(10))
    # model.add(Dense(1))

    # P3 Guy
    # model.add(Conv2D(32, (3, 3), strides=(3, 3), padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(64, (3, 3), strides=(3, 3), padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Conv2D(128, (3, 3), strides=(3, 3), padding='same'))
    # model.add(MaxPooling2D())
    # model.add(Flatten())
    # model.add(Dense(500, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(100, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(10, activation='relu'))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))

model.summary()
if os.path.exists(MODEL_H5):
    print("Load existing model", MODEL_H5)
else:
    print("Create a new model at", MODEL_H5)

optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples),
    epochs=EPOCHS)
model.save(MODEL_H5)

# Plotting
print(history_object.history.keys())
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("{}.result.png".format(MODEL_H5), bbox_inches='tight')
plt.show()