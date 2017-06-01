import tensorflow as tf
import csv
import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D
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

IMG_DIR = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-05-31\\recorded"
DATA_FILE = "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-05-31\\recorded.csv"

STEER_FIELD_ID = 1
SPEED_FIELD_ID = 2

# Data Preparation
MODEL_H5 = os.path.abspath('C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-05-31\\model.h5')
print(MODEL_H5)

lines = []
with open(DATA_FILE) as csvfile:
    reader = csv.reader(csvfile)
    next(reader, None)
    for line in reader:
        lines.append(line)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

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
                    filename = source_path.split('/')[-1]
                    imgpath = os.path.join(IMG_DIR, os.path.basename(filename))
                    image = cv2.imread(imgpath)
                    # Convert to YUV
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
                    images.append(image)

                    steer_from_mid = float(batch_sample[STEER_FIELD_ID])-steer_mid
                    measurement = steer_mid + steer_from_mid + c
                    measurements.append(int(measurement))

                    # Flip
                    image_flipped = np.fliplr(image)
                    # Convert to YUV
                    image_flipped = cv2.cvtColor(image_flipped, cv2.COLOR_BGR2YCrCb)
                    images.append(image_flipped)
                    measurement_flipped = steer_mid - steer_from_mid - c
                    measurements.append(int(measurement_flipped))

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=32)
validation_generator = generator(validation_samples, batch_size=32)

if os.path.exists(MODEL_H5):
    model = load_model(MODEL_H5)
else:
    # Model building
    model = Sequential()
    # YUV Normalization
    model.add(Lambda(
        lambda x: (x - 16) / (np.matrix([235.0, 240.0, 240.0]) - 16) - 0.5,
        input_shape=(TARGET_HEIGHT, TARGET_WIDTH, 3)))
    model.add(Cropping2D(cropping=(())))
    # Dropout setup reference:
    # http://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf
    # Page 1938:
    # Dropout was applied to all the layers of the network with the probability of
    # retaining a hidden unit being p = (0.9, 0.75, 0.75, 0.5, 0.5, 0.5) for the 
    # different layers of the network (going from input to convolutional layers to 
    # fully connected layers).
    model.add(Dropout(0.1))
    model.add(Conv2D(24, (5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))
    model.summary()

optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
    validation_data=validation_generator, validation_steps=len(validation_samples),
    epochs=5)
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