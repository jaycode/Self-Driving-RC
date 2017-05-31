import csv
import numpy as np
import cv2

target_shape = (64, 48)  # original size 640, 480
nb_cams = 1

# Angle to shift from left and right cameras.
angle_shift = 0.1
steer_min = 


def load_csv(directory=''):
    # data_dirs = ['dataset-1', 'dataset-2', 'dataset-3', 'dataset-4', 'dataset-5', 'dataset-water', 'data-udacity']
    data_dirs = ['C:\\Users\\teguh\\Dropbox\\Projects\\Robotics\\Self-Driving-RC-Data\\recorded-2017-05-28']

    global csv_rows_train, csv_rows_val, csv_rows_test
    csv_rows = []

    for dr in data_dirs:
        log_filename = dr + '\\recorded.csv'
        img_path = dr + '\\recorded-2017-05-28\\'

        with open(log_filename, 'r') as csvfile:
            logreader = csv.reader(csvfile)
            next(logreader)
            for row in logreader:
                csv_rows.append(row)
    csv_rows = np.array(csv_rows)

    # split the rows to train test val
    from sklearn.model_selection import train_test_split
    csv_rows_train, csv_rows_test = train_test_split(csv_rows, test_size=0.1)
    csv_rows_train, csv_rows_val = train_test_split(csv_rows_train, test_size=0.1)

    print('CSV loaded, #train = ', len(csv_rows_train),
          '#val = ', len(csv_rows_val),
          '#test = ', len(csv_rows_test))


def load_image(img_filename, angle, images, angles):
    img = cv2.imread(img_filename)
    #print(img_filename)
    #print(img)
    img = cv2.resize(img, target_shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    img = img[:,:,2] #choose S channel
    images.append(img)
    angles.append(float(angle))

def normalize_data(data):
    #X_train = X_train.astype(np.float32)
    data = ((data - data.min())/(np.max(data) - np.min(data)))
    # STOP: Do not change the tests below. Your implementation should pass these tests.
    assert(round(np.mean(data)) == 0), "The mean of the input data is: %f" % np.mean(X_train)
    assert(np.min(data) == 0.0 and np.max(data) == 1.0), "The range of the input data is: %.1f to %.1f" % (np.min(X_train), np.max(X_train))
    return data

def batch_generator(batch_size, source='train'):
    if source == 'train':
        csv_rows = csv_rows_train
    elif source == 'val':
        csv_rows = csv_rows_val
    elif source == 'test':
        csv_rows = csv_rows_test
    else:
        print('Error data segment unknown = ', source)

    row_indices = range(csv_rows.shape[0])

    while 1:
        chosen_indices = np.random.choice(row_indices, size=int(batch_size / nb_cams))
        # print(chosen_indices)
        chosen_rows = csv_rows[chosen_indices]
        # print(chosen_rows.shape)
        images = []
        angles = []
        for row in chosen_rows:
            load_image('/home/sku/recorded/' + row[2].strip(), float(row[0]), images, angles)  # center
            if nb_cams == 3:
                load_image(row[1], float(row[3]) + angle_shift, images, angles)  # left
                load_image(row[2], float(row[3]) - angle_shift, images, angles)  # right

        X = normalize_data(np.array(images).astype('float'))
        X = X.reshape(X.shape[0], X.shape[1], X.shape[2], 1)
        Y = np.array(angles)
        # print('Batch generated.')
        yield (X, Y)


# from https://github.com/Lasagne/Lasagne/issues/12
def threaded_generator(generator, num_cached=10):
    import queue
    queue = queue.Queue(maxsize=num_cached)
    sentinel = object()  # guaranteed unique reference

    # define producer (putting items into queue)
    def producer():
        for item in generator:
            queue.put(item)
        queue.put(sentinel)

    # start producer (in a background thread)
    import threading
    thread = threading.Thread(target=producer)
    thread.daemon = True
    thread.start()

    # run as consumer (read items from queue, in current thread)
    item = queue.get()
    while item is not sentinel:
        yield item
        queue.task_done()
        item = queue.get()


def show_predictions(X_test, Y_test):
    print('Predicting on test set...')
    predictions = model.predict(X_test)
    sum_p = 0
    sum_y = 0
    for i in range(32):
        sum_p += predictions[i][0]
        sum_y += Y_test[i]
        print('p: %.4f' % predictions[i][0], ', y: %.4f' % Y_test[i],
              'c_p: %.4f' % sum_p, ', c_y: %.4f' % sum_y)


from keras.models import Model, Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Convolution2D, MaxPooling2D
from keras.layers.core import Flatten, Dense, Dropout
from keras.optimizers import Adam

model = Sequential()
model.add(Convolution2D(32, 3, 3,
                        border_mode='same',
                        input_shape=(target_shape[1], target_shape[0], 1)))
model.add(MaxPooling2D())
model.add(Convolution2D(64, 3, 3,
                        border_mode='same'))
model.add(MaxPooling2D())
model.add(Convolution2D(128, 3, 3,
                        border_mode='same'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(500, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(1))

my_adam = Adam(lr=0.0001)
model.compile(loss='mse', optimizer=my_adam)

model.summary()

load_csv()

batch_size = nb_cams*int(128/nb_cams) # must be multiply of nb_cams
samples_per_epoch = batch_size * int(len(csv_rows_train) / batch_size)
nb_epoch = 20
nb_val_samples = 5*batch_size

history = model.fit_generator(threaded_generator(batch_generator(batch_size, 'train')),
                    samples_per_epoch, nb_epoch=nb_epoch,
                    verbose=1, validation_data=batch_generator(batch_size, 'val'),
                    nb_val_samples=nb_val_samples)

print('Training completed.')

# Visualize training history
def visualize_history():
    import matplotlib.pyplot as plt
    import numpy

    print(history.history.keys())
    #%matplotlib inline

    # summarize history for loss
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    axes = plt.gca()
    axes.set_ylim([0,np.max(history.history['val_loss'])])
    plt.show()

#visualize_history()

def eval_and_save():
    test_samples = nb_cams*int(1000/nb_cams)
    gen = threaded_generator(batch_generator(test_samples, 'test'))
    X_test, Y_test = next(gen)
    score = model.evaluate(X_test, Y_test, batch_size=batch_size)
    print('Test score (MSE): ', score)

    filepath = 'model-' + '%.4f' % score
    # Save model architecture:
    json_string = model.to_json()
    import json
    with open(filepath + '.json', 'w') as outfile:
        json.dump(json_string, outfile)
    print('Model saved to ', filepath + '.json')
    # save weights:
    model.save_weights(filepath + '.h5')
    print('Weights saved to ', filepath + '.h5')

eval_and_save()
