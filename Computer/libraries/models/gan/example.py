import os, sys
import argparse
import glob
import csv
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Load libraries directory
dir_path = os.path.dirname(os.path.realpath(__file__))
ROOT_DIR = os.path.realpath(os.path.join(dir_path, '..', '..', '..'))
sys.path.append(ROOT_DIR)
from libraries.helpers import configuration, preprocess
from libraries.models import simple_cnn, simple_cnn_generator
from libraries.models.gan import GAN

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
EPOCHS=1
GENERATOR_INPUT_SIZE=100

# ID of center image filename in the csv file.
FILENAME_CENTER_FIELD_ID = 0
STEER_FIELD_ID = 1
SPEED_FIELD_ID = 2

DATA_DIRS = [
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-01.1\\",
# "C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-06-04\\"
"/home/jay/Self-Driving-RC-Data/recorded-2017-06-11-training/"]


def data_generator(samples, batch_size=32):
    steer_range = STEER_MAX - STEER_MIN
    steer_mid = (steer_range/2) + STEER_MIN

    num_samples = len(samples)
    while 1:
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:
                # TODO: Use multiple cameras
                # corrections = [0, (0.2*steer_range), (0.2*steer_range)] # center, left, right
                corrections = [0]

                # The following loop takes data from three cameras: center, left, and right.
                # The steering measurement for each camera is then added by
                # the correction as listed above.
                for i, c in enumerate(corrections):
                    # field number i contains the image.
                    source_path = batch_sample[i]
                    image = cv2.imread(source_path)

                    # Preprocessing
                    image = preprocess(image, TARGET_HEIGHT, TARGET_WIDTH, TARGET_CROP)

                    images.append(image)

                    steer_from_mid = float(batch_sample[STEER_FIELD_ID])-steer_mid
                    measurement = steer_mid + steer_from_mid + c
                    measurements.append(int(measurement))

                    # Flip
                    image_flipped = np.fliplr(image)
                    images.append(image_flipped)
                    measurement_flipped = steer_mid - steer_from_mid - c
                    measurements.append(int(measurement_flipped))

            X_train = np.array(images).astype('float')
            y_train = np.array(measurements)
            yield shuffle(X_train, y_train)

def plot_images(images, rows=4, cols=4):
    plt.figure(figsize=(10,10))
    for i in range(images.shape[0]):
        if i < (rows*cols):
            plt.subplot(rows, cols, i+1)
            image = images[i, :, :, :]
            image = np.reshape(image, [images.shape[1], images.shape[2]])
            plt.imshow(image, cmap='gray')
            plt.axis('off')
    plt.tight_layout()
    plt.show()

    # if save2file:
    #     plt.savefig(filename)
    #     plt.close('all')
    # else:
    #     plt.show()

def main():
    # Preparation
    parser = argparse.ArgumentParser(description='Preprocess images')
    parser.add_argument('dir', type=str,
        help="Where the generated images will be stored.")
    parser.add_argument('--model', type=str,
        default=None,
        help="Path to output model H5 file e.g. `/home/user/model.h5`. Load the file as base if it exists.")
    parser.add_argument('--data-dirs', type=str,
        default=DATA_DIRS,
        nargs='+',
        help="Path to directory that contains csv file and a directory of images.")
    args = parser.parse_args()
    data_dirs = args.data_dirs
    model_h5 = args.model
    if model_h5:
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

    train_generator = data_generator(train_samples, batch_size=BATCH_SIZE)
    validation_generator = data_generator(validation_samples, batch_size=BATCH_SIZE)

    # Model init
    height = (TARGET_HEIGHT-TARGET_CROP[0][0]-TARGET_CROP[0][1])
    width = (TARGET_WIDTH-TARGET_CROP[1][0]-TARGET_CROP[1][1])
    generator_model = simple_cnn(
        height, width, CHANNELS, normalize=NORMALIZE)

    discriminator_model = simple_cnn(
        height, width, CHANNELS, normalize=NORMALIZE)

    discriminator_model = simple_cnn(TARGET_WIDTH, TARGET_HEIGHT, CHANNELS)

    gan = GAN(generator_model, discriminator_model)

    # for training_data in train_generator:
    #     print(training_data[0].shape)
    for i in range(EPOCHS):
        training_data = next(train_generator)
        images_train = training_data[0]
        noise = np.random.uniform(-1.0, 1.0, size=[BATCH_SIZE, 100])
        images_fake = gan.generator.predict(noise)[:, :(TARGET_HEIGHT-TARGET_CROP[0][0]-TARGET_CROP[0][1]), :, :]
        print(images_train.shape)
        print(images_fake.shape)
        x = np.concatenate((images_train, images_fake))
        y = np.ones([2*BATCH_SIZE, 1])
        y[BATCH_SIZE:, :] = 0
        # d_loss = gan.discriminator.train_on_batch(x, y)
        # plot_images(images_fake)
        print(images_fake.shape)
    #     images_train = self.x_train[np.random.randint(0,
    #         self.x_train.shape[0], size=batch_size), :, :, :]
    #     noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    #     images_fake = self.generator.predict(noise)
    #     x = np.concatenate((images_train, images_fake))
    #     y = np.ones([2*batch_size, 1])
    #     y[batch_size:, :] = 0
    #     d_loss = self.discriminator.train_on_batch(x, y)

    #     y = np.ones([batch_size, 1])
    #     noise = np.random.uniform(-1.0, 1.0, size=[batch_size, 100])
    #     a_loss = self.adversarial.train_on_batch(noise, y)
    #     log_mesg = "%d: [D loss: %f, acc: %f]" % (i, d_loss[0], d_loss[1])
    #     log_mesg = "%s  [A loss: %f, acc: %f]" % (log_mesg, a_loss[0], a_loss[1])


    # Callbacks
    # tb = keras.callbacks.TensorBoard(
    #      log_dir=os.path.join(model_dir, 'graph'), histogram_freq=0,  
    #      write_graph=True, write_images=True)
    # es = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=0, mode='auto')



    # if model_h5 and os.path.exists(model_h5):
    #     print("Load existing model", model_h5)
    # else:
    #     print("Create a new model at", model_h5)

    # optimizer = Adam()
    # model.compile(loss='mse', optimizer=optimizer)
    # history_object = model.fit_generator(train_generator, steps_per_epoch=len(train_samples),
    #     validation_data=validation_generator, validation_steps=len(validation_samples),
    #     epochs=EPOCHS, callbacks=[tb, es])
    # model.save(model_h5)


if __name__ == "__main__":
    main()
