import os
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Conv2D, Dropout, Activation, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import UpSampling2D, Conv2DTranspose

def small_cnn(target_height, target_width, channels, model_h5=None, normalize=True):
    model = None
    if model_h5 and os.path.exists(model_h5):
        model = load_model(model_h5)
    else:
        # Model building
        model = Sequential()
        if normalize:
            model.add(Lambda(lambda x: x/255.0,
                input_shape=(target_height, target_width, channels)))
        else:
            model.add(Lambda(lambda x: x/1.0,
                input_shape=(target_height, target_width, channels)))

        # YUV Normalization
        # model.add(Lambda(
        # lambda x: (x - 16) / (np.matrix([235.0, 240.0, 240.0]) - 16) - 0.5,
        # input_shape=(target_height,target_width, 3)))

        dropout = 0.4
        dropout_fconn = 0.1

        # 320x110x1 (assuming 1 channel)
        model.add(Conv2D(24, (5, 5), strides=(1,2), activation='relu', padding='same'))
        model.add(Dropout(dropout))
        # 160x110x24
        model.add(Conv2D(36, (5, 5), strides=(2,2), activation='relu', padding='same'))
        model.add(Dropout(dropout))
        # 80x55x36
        model.add(Conv2D(48, (5, 5), strides=(2,2), activation='relu', padding='same'))
        model.add(Dropout(dropout))
        # 40x28x48
        model.add(Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
        model.add(Dropout(dropout))
        # 20x14x64
        model.add(Conv2D(64, (3, 3), strides=(2,2), activation='relu', padding='same'))
        model.add(Dropout(dropout))
        # 10x7x64 = 4480
        model.add(Flatten())
        model.add(Dense(10))
        model.add(Dropout(dropout_fconn))
        model.add(Dense(1))
       
    return model

def small_cnn_generator(target_height, target_width, channels, model_h5=None):
    model = None
    if model_h5 and os.path.exists(model_h5):
        model = load_model(model_h5)
    else:
        dropout = 0.4
        model = Sequential()
        # Input to conv layers should be 7x10x64 (see simple_cnn.py)
        model.add(Dense(7*10*64, input_dim=100))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(Reshape((7, 10, 64)))
        model.add(Dropout(dropout))

        # 7x10x64
        model.add(Conv2DTranspose(64, 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))

        # 7x10x64
        model.add(Conv2DTranspose(64, 3, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # 14x20x64
        model.add(Conv2DTranspose(48, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # 28x40x48
        model.add(Conv2DTranspose(36, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(UpSampling2D())

        # 56x80x36
        model.add(Conv2DTranspose(24, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(UpSampling2D())
        
        # 112x160x24
        model.add(Conv2DTranspose(1, 5, padding='same'))
        model.add(BatchNormalization(momentum=0.9))
        model.add(Activation('relu'))
        model.add(UpSampling2D((1, 2)))

        # 112x320x1
        model.add(Activation('sigmoid'))
        model.summary()
    return model