import os

from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Flatten, Dense, Lambda
from keras.models import Sequential, load_model


# Tiny model
# Total params: 136,937

def build(target_height, target_width, channels, model_h5=None, normalize=True):
    model = None
    if model_h5 and os.path.exists(model_h5):
        model = load_model(model_h5)
    else:
        # Model building
        model = Sequential()
        if normalize:
            model.add(Lambda(lambda x: x / 255.0,
                             input_shape=(target_height, target_width, channels)))
        else:
            model.add(Lambda(lambda x: x / 1.0,
                             input_shape=(target_height, target_width, channels)))

        # 160x60x1 (assuming 1 channel)
        model.add(Conv2D(16, (5, 5), activation='relu', padding='same', use_bias=True))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 80x60x16
        model.add(Conv2D(24, (5, 5), activation='relu', padding='same', use_bias=True))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 40x30x24
        model.add(Conv2D(32, (5, 5), activation='relu', padding='same', use_bias=True))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 20x10x32
        model.add(Conv2D(48, (5, 5), activation='relu', padding='same', use_bias=True))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        # 10x5x48 = 2400
        model.add(Flatten())
        model.add(Dense(48, activation='relu', use_bias=True))
        model.add(Dense(1))

    return model