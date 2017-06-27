import numpy as np
import cv2
import h5py
import serial
from keras.models import load_model
import os, sys
import pickle
from PIL import Image, ImageDraw

dir_path = os.path.dirname(os.path.realpath(__file__))

# Path to Computer root directory
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..'))

sys.path.append(ROOT_DIR)
from libraries.find_lines_sliding_windows import FindLinesSlidingWindows
from libraries.line_helpers import annotate_with_lines

def configuration():
    """ The car's configuration.

    These are the settings that are used across scripts.
    """
    return {
        'target_width': 320,
        'target_height': 240,
        'target_crop': ((120, 10), (0, 0)), # (top, bottom), (left, right)
        'target_scale': 0.5,

        'poly_target_width': 320,
        'poly_target_height': 240,
        'poly_target_scale': 1.0,
        'poly_channels': 3,

        # Look into Arduino code's car.h for SteerFeedMin_ and SteerFeedMax_
        'steer_min': 0,
        'steer_max': 1023,
        # Number of image channels.
        'channels': 1,
        # If the values in image has values 0 to 255, set to True.
        'normalize': False,
        'preprocessed_dir': 'preprocessed'
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

def prepare_initial_transformation(calibration_path, img_height, img_width, scale=1.0):
    # === Perspective Transformation ===
    img_height *= scale
    img_width *= scale

    top_width = img_width
    left_top = (img_width - top_width) / 2
    right_top = left_top + top_width
    bottom_width = img_width*3
    left_bottom = (img_width - bottom_width) / 2
    right_bottom = left_bottom + bottom_width
    top = 115*scale
    bottom = 180*scale

    src = np.float32([[left_bottom,bottom],
                     [left_top,top],
                     [right_top,top],
                     [right_bottom,bottom]])

    width = img_width
    left = (img_width - width) / 2
    right = left + width
    top = 0
    bottom = img_height
    dst = np.float32([[left,bottom],
                     [left,top],
                     [right,top],
                     [right,bottom]])

    M = cv2.getPerspectiveTransform(src, dst)
    Minv = cv2.getPerspectiveTransform(dst, src)

    # === END Perspective Transformation ===

    # === Calibration ===
    with open( calibration_path, "rb" ) as pfile:
        cal = pickle.load(pfile)
    mtx = cal['mtx']
    dist = cal['dist']
    # === END Calibration ===

    return (mtx, dist, M, Minv)

# Unused: This was for behavioral cloning
def preprocess_behavioral_cloning(img_raw, height, width, crop):
    """ Preprocess images before doing behavioral cloning.

    Make sure to change "channels" in configuration() function to follow suit.
    Cropping needs to be done here to allow for GAN.
    """
    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 1]
    img = cv2.Sobel(img, -1, 1, 0, ksize=3)
    img = img / 255.0
    img = img > 0.5
    
    img1 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 2]
    img1 = cv2.Sobel(img1, -1, 0, 1, ksize=3)
    img1 = img1 / 255.0
    img1 = img1 > 0.5

    img_final = (img==1) | (img1==1)
    img_final = np.array([img_final])
    img_final = np.rollaxis(np.concatenate((img_final, img_final, img_final)), 0, 3)
    return img_final[crop[0][0]:(height-crop[0][1]), crop[1][0]:(width-crop[1][1]), [0]]

# Unused: Too slow
def preprocess_and_find_lines(img_raw, height, width, crop, mtx, dist, M, Minv):
    print("img_raw shape", img_raw.shape)

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 1]
    img = cv2.Sobel(img, -1, 1, 0, ksize=3)
    img = img > 127
    
    img1 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 2]
    img1 = cv2.Sobel(img1, -1, 0, 1, ksize=3)
    img1 = img1 > 127
    
    img2 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
    img2 = cv2.inRange(img2, (38, 61, 112), (139, 255, 255))
    img2 = img2 > 25.5

    final_img = (img==1) | (img1==1) | (img2==1)
    
    f3 = np.stack((final_img, final_img, final_img), axis=2)
    f3 = (f3 * 255.0).astype(np.uint8)

    undist = cv2.undistort(f3, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(undist, M, (width, height))
    warped = warped[:, :, 0]
    print("after first warp",warped.shape)

    finder = FindLinesSlidingWindows(nwindows=30, window_minpix=20, subsequent_search_margin=30,
                                     lr_start_slack=0.1, debug=False, error_top_percentile=75,
                                     center_importance=4,
                                     error_threshold=30, window_patience=7, window_empty_px=5)
    fits = finder.process(warped)
    img_final = annotate_with_lines(warped, list(map(lambda fit: fit[0], fits)), keep_background=False)
    print("img_final before last warp", img_final.shape)

    img_final = cv2.warpPerspective(img_final, Minv, (img_final.shape[1], img_final.shape[0]))
    # cv2.imshow('img', img_final)
    # cv2.waitKey()

    print("img_final shape", img_final.shape)
    return img_final[crop[0][0]:(height-crop[0][1]), crop[1][0]:(width-crop[1][1]), [0]]

# def preprocess_line_finding(img_raw, mtx, dist, M, inrange=((38, 61, 112), (139, 255, 255))):
#     """ Preprocess before line finding.
#     """
#     img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 1]
#     img = cv2.Sobel(img, -1, 1, 0, ksize=3)
#     img = img > 127
    
#     img1 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 2]
#     img1 = cv2.Sobel(img1, -1, 0, 1, ksize=3)
#     img1 = img1 > 127
    
#     img2 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
#     img2 = cv2.inRange(img2, (38, 61, 112), (139, 255, 255))
#     img2 = img2 > 25.5

#     final_img = (img==1) | (img1==1) | (img2==1)
    
#     f3 = np.stack((final_img, final_img, final_img), axis=2)
#     f3 = (f3 * 255.0).astype(np.uint8)

#     undist = cv2.undistort(f3, mtx, dist, None, mtx)
#     warped = cv2.warpPerspective(undist, M, (img_raw.shape[1], img_raw.shape[0]))
#     return warped[:, :, 0]

def preprocess_line_finding(img_raw, M, scale=1.0, mtx=None, dist=None, sobel=True, inrange=[[38, 61, 112], [139, 255, 255]]):
    """ Preprocess before line finding.
    """
    img_raw = resize_image_by_pil(img_raw, scale)

    img = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 1]
    img = cv2.Sobel(img, -1, 1, 0, ksize=3)
    img = img > 127
    
    img1 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)[:, :, 2]
    img1 = cv2.Sobel(img1, -1, 0, 1, ksize=3)
    img1 = img1 > 127
    
    img2 = cv2.cvtColor(img_raw, cv2.COLOR_BGR2HSV)
    img2 = cv2.inRange(img2, tuple(inrange[0]), tuple(inrange[1]))
    img2 = img2 > 25.5

    if sobel:
        final_img = (img==1) | (img1==1) | (img2==1)
    else:
        final_img = (img2==1)
    
    f3 = np.stack((final_img, final_img, final_img), axis=2)
    f3 = (f3 * 255.0).astype(np.uint8)

    if mtx is not None and dist is not None:
        f3 = cv2.undistort(f3, mtx, dist, None, mtx)
    warped = cv2.warpPerspective(f3, M, (img_raw.shape[1], img_raw.shape[0]))
    return warped[:, :, 0]

def load_from(data_path):
    """ Load file into list.
    """
    data = []
    buffersize = 2**16
    with open(data_path) as f: 
        while True:
            lines_buffer = f.readlines(buffersize)
            if not lines_buffer:
                break
            for line in lines_buffer:
                data.append(line.strip())
    return data

def remove_generic(path, __func__):
    try:
        __func__(path)
    except OSError as error:
        print("OS error: {0}".format(error))

def clean_dir(path):
    if not os.path.isdir(path):
        return

    files = os.listdir(path)
    for x in files:
        full_path = os.path.join(path, x)
        if os.path.isfile(full_path):
            f = os.remove
            remove_generic(full_path, f)
        elif os.path.isdir(full_path):
            clean_dir(full_path)
            f = os.rmdir
            remove_generic(full_path, f)


def convert_rgb_to_y(image, jpeg_mode=True, max_value=255):
    if len(image.shape) <= 2 or image.shape[2] == 1:
        return image

    y_image = np.zeros([image.shape[0], image.shape[1], 1])  # type: np.ndarray
    if jpeg_mode:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                y_image[i, j, 0] = 0.299 * image[i, j, 0] + 0.587 * image[i, j, 1] + 0.114 * image[i, j, 2]
    else:
        for i in range(image.shape[0]):
            for j in range(image.shape[1]):
                y_image[i, j, 0] = (16.0 * max_value + 65.481 * image[i, j, 0] + 128.553 * image[i, j, 1] + 24.966 *
                                    image[i, j, 2]) / 256.0
    return y_image


def resize_image_by_pil(image, scale, resampling_method="bicubic"):
    width, height = image.shape[1], image.shape[0]
    new_width = int(width * scale)
    new_height = int(height * scale)

    if resampling_method == "bicubic":
        method = Image.BICUBIC
    elif resampling_method == "bilinear":
        method = Image.BILINEAR
    elif resampling_method == "nearest":
        method = Image.NEAREST
    elif resampling_method == "hamming":
        method = Image.HAMMING
    elif resampling_method == "box":
        method = Image.BOX
    else:
        method = Image.LANCZOS

    if len(image.shape) == 3 and image.shape[2] == 3:
        image = Image.fromarray(image, "RGB")
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
    else:
        image = Image.fromarray(image.reshape(height, width))
        image = image.resize([new_width, new_height], resample=method)
        image = np.asarray(image)
        image = image.reshape(new_height, new_width, 1)
    return image