"""
Trying to learn with Deep-Learning
Now: Testing data for learning
"""

import argparse
import csv
import glob
import os
import pickle
import sys
import pdb

import cv2
import keras
import matplotlib.pyplot as plt
import numpy as np
from keras.optimizers import Adam
from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
import libraries.models.tiny_cnn_polynet as models
from libraries.helpers import configuration, preprocess_line_finding, clean_dir, prepare_initial_transformation

print("keras version:", keras.__version__)

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)

config = configuration()

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = config['poly_target_width']
TARGET_HEIGHT = config['poly_target_height']
TARGET_SCALE = config['poly_target_scale']
CHANNELS = config['poly_channels']
NORMALIZE = config['normalize']
PRE_PROCESSED_DIR = config['preprocessed_dir']

BATCH_SIZE = 32
EPOCHS = 40

DATA_DIRS = []

CALIBRATION_FILE = os.path.realpath(os.path.join(dir_path, '..', 'calibrations', 'cal-elp.p'))

# ID of center image filename in the csv file.
FILENAME_ID = 0
LEFT_HASLINE_ID = 1
LEFT_COEF1_ID = 2
LEFT_COEF2_ID = 3
LEFT_COEF3_ID = 4
RIGHT_HASLINE_ID = 5
RIGHT_COEF1_ID = 6
RIGHT_COEF2_ID = 7
RIGHT_COEF3_ID = 8

# The model is going to be created at this path. If the model
# already exists at that location, use it as the base for subsequent learning.
# MODEL_H5 = os.path.abspath('C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\model.h5')
MODEL_H5 = os.path.abspath('/home/jay/Self-Driving-RC-Data/model.h5')

def main():
    mtx, dist, M, Minv = prepare_initial_transformation(
        CALIBRATION_FILE, TARGET_HEIGHT, TARGET_WIDTH)

    parser = argparse.ArgumentParser(description='Learn')
    parser.add_argument('model', type=str,
                        default=MODEL_H5,
                        help="Path to output model H5 file e.g. `/home/user/polyfit.h5`. Load the file as base if it exists.")
    parser.add_argument('--data-dirs', type=str,
                        default=DATA_DIRS,
                        nargs='+',
                        help="Path to directory that contains csv file and a directory of images. CSV name must be poly.csv and " + \
                             "images directory name should be `recorded/`")
    parser.add_argument('--data-num', type=int,
                        default=0,
                        help="Number of data to process.")
    args = parser.parse_args()
    data_dirs = args.data_dirs
    data_num = args.data_num
    model_h5 = args.model
    model_dir = os.path.dirname(model_h5)
    os.makedirs(model_dir, exist_ok=True)

    x_height = int((TARGET_HEIGHT) * TARGET_SCALE)
    x_width = int((TARGET_WIDTH) * TARGET_SCALE)

    lines = []
    for i, data_dir in enumerate(data_dirs):
        # Get the first csv file in the dir.
        data_file = glob.glob(os.path.join(data_dir, "poly.csv"))[0]
        print("Get data from", data_file)
        with open(data_file) as csvfile:
            reader = csv.reader(csvfile)
            next(reader, None)
            lines_read = 0
            for line in reader:
                if data_num > 0 and data_num <= lines_read:
                    break
                # Convert filename to complete path
                imgpath = os.path.abspath(os.path.join(data_dir, "recorded", line[FILENAME_ID]))
                line[FILENAME_ID] = imgpath
                lines.append(line)
                lines_read += 1
            print("Number of rows read:", lines_read)

    tb = keras.callbacks.TensorBoard(
        log_dir=os.path.join(model_dir, 'graph'), histogram_freq=1,
        write_graph=True, write_images=False)
    es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')

    model = models.build(x_height, x_width, CHANNELS, model_h5=model_h5, normalize=NORMALIZE)
    model.summary()

    if os.path.exists(model_h5):
        print("Load existing model", model_h5)
    else:
        print("Create a new model at", model_h5)

    clean_dir(os.path.join(model_dir, 'graph'))

    optimizer = Adam()
    model.compile(loss='mse', optimizer=optimizer)

    # allocating memories
    training_count = len(lines)
    x_train = np.zeros([training_count, x_height, x_width, CHANNELS])
    # Number of output tensors is applied here.
    y_train = np.zeros([training_count, 4])

    for i in range(training_count):
        image = misc.imread(lines[i][FILENAME_ID])
        x_train[i] = image
        y_train[i] = [
            lines[i][LEFT_HASLINE_ID],
            lines[i][LEFT_COEF1_ID],
            lines[i][LEFT_COEF2_ID],
            lines[i][LEFT_COEF3_ID]
        ]
        # y_train[i] = lines[i][LEFT_COEF1_ID]

    # checking gt distribution
    plt.hist(y_train)
    plt.savefig("gt_hist.png")

    plt.clf()
    plt.plot(y_train)
    plt.savefig("gt_plot.png")

    history_object = model.fit(x=x_train, y=y_train, batch_size=BATCH_SIZE, epochs=EPOCHS,
                               verbose=1, shuffle=True, validation_split=0.2, callbacks=[tb, es])

    model.save(model_h5)

    # Plotting
    with open(model_h5.replace('.h5', '-history.pkl'), 'wb') as handle:
        pickle.dump(history_object.history, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.clf()
    plt.plot(history_object.history['loss'])
    plt.plot(history_object.history['val_loss'])
    plt.title('model mean squared error loss')
    plt.ylabel('mean squared error loss')
    plt.xlabel('epoch')
    plt.legend(['training set', 'validation set'], loc='upper right')
    plt.savefig("{}.result.png".format(model_h5), bbox_inches='tight')
    plt.show()
    
if __name__ == "__main__":
    main()