# General Adversarial Network.
# Run `example.py` script to show some example generated images.
import os, sys
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping1D, Cropping2D
from keras.layers import Conv2D, Dropout, BatchNormalization
from keras.layers.pooling import MaxPooling2D

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