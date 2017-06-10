# Test loading keras model

import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping1D, Cropping2D
from keras.layers import Conv2D, Dropout
from keras.layers.pooling import MaxPooling2D
from keras.optimizers import Adam
import numpy as np
import csv
from sklearn.utils import shuffle
import os
import cv2
import matplotlib.pyplot as plt

import pickle

MODEL_H5 = '/home/sku/model.h5'

model = load_model(MODEL_H5)