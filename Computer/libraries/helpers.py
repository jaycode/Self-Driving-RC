import numpy as np
import cv2
import h5py
import serial
from keras.models import load_model

def configuration():
    """ The car's configuration.

    These are the settings that are used across scripts.
    """
    return {
        'target_width': 320,
        'target_height': 240,
        'target_crop': ((60, 20), (0, 0)) # (top, bottom), (left, right)
    }

def choose_port(ports):
    """ Find a serial port that connects to arduino.
    """
    port_connected = False
    port_idx = 0
    while not port_connected:
        try:
            port = serial.Serial(ports[port_idx], baudrate=9600, timeout=0.05)
            port_connected = True
            print("Port {} connected!\n".format(ports[port_idx]))
        except:
            print("Port {} not connected".format(ports[port_idx]))
            port_idx+=1;
            if len(ports) > port_idx:
                print(", trying {}...".format(ports[port_idx]))
            else:
                print("No other port to try.")
                exit();
    return port

def prepare_model(model_path):
    """ Loads a model.
    """
    from keras import __version__ as keras_version
    # check that model Keras version is same as local Keras version
    f = h5py.File(model_path, mode='r')
    model_version = f.attrs.get('keras_version')
    keras_version = str(keras_version).encode('utf8')

    if model_version != keras_version:
        print('You are using Keras version ', keras_version,
              ', but the model was built using ', model_version)

    # Load model routine may generate error due to incompatible python
    # compiler used to generate model.h5 file:
    # - "Segmentation fault (core dumped)": h5 file created with python 3.6, drive.py uses python 3.5.
    # - "SystemError: unknown opcode": h5 file created with python 3.5, drive.py uses python 3.6.
    return load_model(model_path)

def preprocess(raw_img):
    """ Preprocess images.

    Image outputted by this function should be converted as follows:
    ```
    img = preprocess(raw_image)
    img_array = np.asarray(img)[None, :, :, :]
    ```
    before being used as input by the model.

    """
    img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2HSV)[:, :, 2]
    img = cv2.Sobel(img, -1, 0, 1, ksize=3)
    img = img / 255.0
    img = img > 0.5
    img = np.array([img])
    img = np.rollaxis(np.concatenate((img, img, img)), 0, 3)
    return img[:, :, [0]]