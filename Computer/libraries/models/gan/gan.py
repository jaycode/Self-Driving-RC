# General Adversarial Network training.
# Run `example.py` script to show some example generated images.
import os, sys
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping1D, Cropping2D
from keras.layers import Conv2D, Dropout, BatchNormalization
from keras.layers.pooling import MaxPooling2D

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, preprocess
from libraries.models import simple_cnn

config = configuration()

# Make sure the target shape is the same with the one in driver/main.py
# i.e. look for cams setup with variable CAP_PROP_FRAME_WIDTH and CAP_PROP_FRAME_HEIGHT.
TARGET_WIDTH = config['target_width']
TARGET_HEIGHT = config['target_height']
TARGET_CROP = config['target_crop']
STEER_MIN = config['steer_min']
STEER_MAX = config['steer_max']
CHANNELS = config['channels']

class GAN(object):
    def __init__(self, generator, discriminator, model_h5=None):
        self.model_h5 = model_h5
        self.discriminator = self.setup_discriminator(discriminator)
        self.generator = generator
        self.adversarial = self.setup_adversarial(generator, discriminator)

    def setup_discriminator(self, discriminator):
        """ Prepare discriminator for training
        """
        optimizer = RMSprop(lr=0.0002, decay=6e-8)
        model = Sequential()
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return model

    def setup_adversarial(self, generator, discriminator):
        """ Prepare adversarial model for training
        """
        optimizer = RMSprop(lr=0.0001, decay=3e-8)
        model = Sequential()
        model.add(generator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return model