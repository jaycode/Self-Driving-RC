import sys
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
import pdb

import keras
from keras.optimizers import Adam

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, preprocess1
from libraries.models import simple_cnn, small_cnn

config = configuration()

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']
STEER_MIN = config['steer_min']
STEER_MAX = config['steer_max']
CHANNELS = config['channels']
NORMALIZE = config['normalize']

BATCH_SIZE=32
EPOCHS=40

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

parser = argparse.ArgumentParser(description='Learn')
parser.add_argument('model', type=str,
    default=MODEL_H5,
    help="Path to output model H5 file e.g. `/home/user/model.h5`. Load the file as base if it exists.")
parser.add_argument('--data-dirs', type=str,
    default=DATA_DIRS,
    nargs='+',
    help="Path to directory that contains csv file and a directory of images.")
args = parser.parse_args()
data_dirs = args.data_dirs
model_h5 = args.model
model_dir = os.path.dirname(model_h5)
os.makedirs(model_dir, exist_ok=True)

lines = []
csv_files = data_dirs
for i, data_dir in enumerate(data_dirs):
    # Get the first csv file in the dir.
    data_file = glob.glob(os.path.join(data_dir, '*.csv'))[0]
    print("Get data from", data_file)
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
        print("Number of rows now", len(lines))

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# def preprocess(raw_img):
#     img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)[:, :, 2]
#     img = cv2.Sobel(img, -1, 0, 1, ksize=3)
#     img = img / 255.0
#     img = img > 0.5
#     img = np.array([img])
#     img = np.rollaxis(np.concatenate((img, img, img)), 0, 3)
#     return img[:, :, [0]]

def generator(samples, batch_size=32):
    steer_range = STEER_MAX - STEER_MIN
    steer_mid = (steer_range/2) + STEER_MIN

    num_samples = len(samples)
    while 1:
        shuffle(samples)
        # pdb.set_trace()
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                
                # field number 0 contains the center image.
                source_path = batch_sample[0]
                image = cv2.imread(source_path)

                # Preprocessing
                image = preprocess1(image, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CROP)

                images.append(image)

                steer_from_mid = float(batch_sample[STEER_FIELD_ID])
                measurement = steer_mid + steer_from_mid
                measurements.append(int(measurement))

                # Flip
                image_flipped = np.fliplr(image)
                images.append(image_flipped)
                measurement_flipped = steer_mid - steer_from_mid
                measurements.append(int(measurement_flipped))

            X_train = np.array(images).astype('float')
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

train_generator = generator(train_samples, batch_size=BATCH_SIZE)
validation_generator = generator(validation_samples, batch_size=BATCH_SIZE)

tb = keras.callbacks.TensorBoard(
     log_dir=os.path.join(model_dir, 'graph'), histogram_freq=0,  
     write_graph=True, write_images=True)
es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

model = simple_cnn(
    (TARGET_HEIGHT-TARGET_CROP[0][0]-TARGET_CROP[0][1]),
    (TARGET_WIDTH-TARGET_CROP[1][0]-TARGET_CROP[1][1]),
    CHANNELS, model_h5=model_h5, normalize=NORMALIZE)
model.summary()

if os.path.exists(model_h5):
    print("Load existing model", model_h5)
else:
    print("Create a new model at", model_h5)

optimizer = Adam()
model.compile(loss='mse', optimizer=optimizer)
history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples)//BATCH_SIZE,
    validation_data=validation_generator, validation_steps=len(validation_samples)//BATCH_SIZE,
    epochs=EPOCHS, callbacks=[tb, es])
model.save(model_h5)

# gan = GAN()
# gan.model_h5 = model_h5
# gan.target_width = TARGET_WIDTH
# gan.
# learner.train()
# model = learner.model

# Plotting
with open(model_h5.replace('.h5', '-history.pkl'), 'wb') as handle:
    pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.savefig("{}.result.png".format(model_h5), bbox_inches='tight')
plt.show()