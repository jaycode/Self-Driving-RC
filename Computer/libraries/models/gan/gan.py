# General Adversarial Network.
# Run `example.py` script to show some example generated images.
import os, sys
from keras.models import Sequential
from keras.optimizers import RMSprop

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
        print("disc is", discriminator)
        model.add(discriminator)
        model.compile(loss='binary_crossentropy', optimizer=optimizer,\
            metrics=['accuracy'])
        return model