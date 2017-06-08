import pandas as pd
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse

from keras.models import Sequential, Model
from keras.layers import Flatten, Dense, Dropout, Flatten, Lambda, Cropping2D, Input
from keras.layers import Conv2D, MaxPooling2D

from keras.applications.vgg16 import VGG16

DATA_DIR = '../data/behavioral-cloning-data/data'


def load_data():
    image_dir = os.path.join(DATA_DIR, 'IMG')
    log_csv = os.path.join(DATA_DIR, 'driving_log.csv')
    log_df = pd.read_csv(log_csv, names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])

    X = np.array([cv2.imread(x) for x in log_df['center']], dtype=np.uint8)
    y = log_df['steering'].as_matrix()
    return X, y


def augment(X, y):
    flipped_X = np.array([cv2.flip(i, 1) for i in X])
    flipped_y = -1.0 * y
    aug_X = np.concatenate([X, flipped_X])
    aug_y = np.concatenate([y, flipped_y])
    return aug_X, aug_y


def training(X, y, modelname='model.h5', gpu=-1):
    if gpu < 0:
        print("Using CPU and LeNet...")
        model = Sequential()
        model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=X.shape[1:]))
        model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')) #, dim_ordering='tf'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
    else:
        print("Using GPU and LeNet...")

        # input_tensor = Input(shape=X.shape[1:])
        in_model = Sequential()
        in_model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=X.shape[1:]))
        in_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=in_model.output)

        fc_model = Sequential()
        fc_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        fc_model.add(Dense(512, activation='relu', kernel_initializer='he_normal'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(128, activation='relu', kernel_initializer='he_normal'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(1, name='output', kernel_initializer='he_normal'))
        
        model = Model(input=vgg16_model.input, output=fc_model(vgg16_model.output))
        
        # model.add(Flatten())
        # model.add(Dense(128, activation='relu'))
        # model.add(Dropout(0.5))
        # model.add(Dense(1))

    model.compile(loss='mse', optimizer='adam')
    model.fit(X, y, validation_split=0.2, shuffle=True, epochs=10)
    model.save(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU flag')
    args = parser.parse_args()
    X, y = load_data()
    X, y = augment(X, y)
    training(X, y, modelname='model.h5', gpu=args.gpu)



"""
history_object = model.fit_generator(train_generator, samples_per_epoch =
    len(train_samples), validation_data = 
    validation_generator,
    nb_val_samples = len(validation_samples), 
    nb_epoch=5, verbose=1)

### print the keys contained in the history object
print(history_object.history.keys())

### plot the training and validation loss for each epoch
plt.plot(history_object.history['loss'])
plt.plot(history_object.history['val_loss'])
plt.title('model mean squared error loss')
plt.ylabel('mean squared error loss')
plt.xlabel('epoch')
plt.legend(['training set', 'validation set'], loc='upper right')
plt.show()
"""