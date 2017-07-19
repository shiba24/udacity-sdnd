import pandas as pd
import numpy as np
import os
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import copy

import keras
from keras.models import Sequential, Model

from keras.layers import Flatten, Dense, Dropout, Flatten, Lambda, Cropping2D, Input
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

from keras.applications.vgg16 import VGG16

DATA_DIR = '../data/behavioral-cloning-data'
DELTA_ANGLE = 0.1

class MiniBatchLoader(object):
    def __init__(self, data_dir, batchsize, insize=(100, 200), train=True):
        self.data_dir = data_dir
        self.batchsize = batchsize
        self.insize = self.get_insize(insize)
        self.train = train
        self.split_train_test()

    def get_insize(self, insize):
        if insize is None:
            return self.calc_inseize()
        else:
            return insize

    def load_csv(self):
        image_dir = os.path.join(self.data_dir, 'IMG')
        log_csv = os.path.join(self.data_dir, 'driving_log_additional.csv')
        self.log_df = pd.read_csv(log_csv) #, names=['center', 'left', 'right', 'steering', 'throttle', 'break', 'speed'])

    def load_csv_path(self, img_path):
        if img_path[0] == '/':
            return os.path.join(DATA_DIR, img_path[63:])
        else:
            return os.path.join(DATA_DIR, img_path)

    def split_train_test(self, split_ratio=.9):
        self.load_csv()
        all_X_list = self.log_df['center']
        all_y_list = self.log_df['steering'].as_matrix()

        all_X_list.append(self.log_df['left'])
        all_X_list.append(self.log_df['right'])
        all_y_list = np.concatenate([all_y_list,
                                     self.log_df['steering'].as_matrix() + DELTA_ANGLE, 
                                     self.log_df['steering'].as_matrix() - DELTA_ANGLE])

        self.datasize = len(all_X_list)
        self.datasize_train = int(self.datasize * split_ratio)
        self.datasize_test = self.datasize - self.datasize_train
        print("training datasets: ", self.datasize_train, "test datasets: ", self.datasize_test)

        indices = np.random.permutation(self.datasize)
        self.train_X_file_list = [self.load_csv_path(all_X_list[indices[i]]) for i in range(0, self.datasize_train)]
        self.train_y_list = np.array([all_y_list[indices[i]] for i in range(0, self.datasize_train)])
        self.test_X_file_list = [self.load_csv_path(all_X_list[indices[i]]) for i in range(self.datasize_train, self.datasize)]
        self.test_y_list = np.array([all_y_list[indices[i]] for i in range(self.datasize_train, self.datasize)])


    def __iter__(self):   # iterator setting
        return self

    # initialize for each training loop
    def initialize_iterator(self):
        self.current_index = 0
        if self.train:
            self.random_index = np.random.permutation(self.datasize_train)
        else:
            self.random_index = np.random.permutation(self.datasize_test)

    def next(self):       # for each loop
        if self.train:
            try:
                _ = self.current_index + 1
            except AttributeError:
                print("Create Iterator settings")
                self.initialize_iterator()
            finally:
                ind_Xy = self.random_index[self.current_index:self.current_index + self.batchsize]
                # make minibatch
                minibatch_path_X = [self.train_X_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_X = self.load_batch_X(minibatch_path_X)
                minibatch_y = self.train_y_list[ind_Xy[0:self.batchsize]]
                minibatch_X, minibatch_y = self.process_batch(minibatch_X, minibatch_y)

                self.current_index += self.batchsize
                if self.current_index + self.batchsize > self.datasize_train:
                    # del self.current_index, self.random_index  # for try-catch
                    # raise StopIteration
                    self.initialize_iterator()
                return minibatch_X, minibatch_y
        else:
            try:
                _ = self.current_index + 1
            except AttributeError:
                print("Create Iterator settings")
                self.initialize_iterator()
            finally:
                ind_Xy = self.random_index[self.current_index:self.current_index + self.batchsize]
                # make minibatch
                minibatch_path_X = [self.test_X_file_list[ind_Xy[i]] for i in range(0, self.batchsize)]
                minibatch_X = self.load_batch_X(minibatch_path_X)
                minibatch_y = self.test_y_list[ind_Xy[0:self.batchsize]]
                minibatch_X, minibatch_y = self.process_batch(minibatch_X, minibatch_y)

                self.current_index += self.batchsize
                if self.current_index + self.batchsize > self.datasize_test:
                    # del self.current_index, self.random_index  # for try-catch
                    # raise StopIteration
                    self.initialize_iterator()
                return minibatch_X, minibatch_y

    # apply for minibatch
    def load_batch_X(self, minibatch_path_X):
        return np.array([cv2.imread(x) for x in minibatch_path_X], dtype=np.uint8)

    def crop_under(self, minibatch_X):
        return minibatch_X[:, 60:-20, :, :]

    def process_batch(self, minibatch_X, minibatch_y):
        minibatch_X = self.crop_under(minibatch_X)
        if self.train:
            delta_hue = np.random.uniform(-18, 18, (minibatch_X.shape[0])).astype(np.int8)            # in opencv, hue is [0, 179]
            processed_X = np.array([self.change_hue(minibatch_X[i, :, :, :], delta_hue[i]) for i in range(len(minibatch_X))])
            processed_X, processed_y = self.fliplr(processed_X, minibatch_y)
            # processed_y = np.random.normal(0., 0.01, size=(len(minibatch_y, )))
        else:
            processed_X = minibatch_X
            processed_y = minibatch_y
        # reshaped_X = np.transpose(self.standardize(processed_X), (0, 3, 1, 2))        # n_batch, n_channel, h, w        
        return processed_X, processed_y

    # apply for each image
    def change_hue(self, img, delta_hue):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[:, :, 0] += delta_hue
        hued_img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
        return hued_img

    def fliplr(self, minibatch_X, minibatch_y):
        def flip(img, flag):
            if flag == -1:
                return cv2.flip(img, 1)
            else:
                return img
        flipflag = np.random.choice((-1, 1), size=(len(minibatch_X), ))   # -1... flip  /  1...not flip
        flipped_X = np.array([flip(minibatch_X[i], flipflag[i]) for i in range(0, len(minibatch_X))])
        flipped_y = flipflag * minibatch_y
        return flipped_X, flipped_y

    def standardize(self, images, mean_image="mean.jpg"):
        if not os.path.exists(mean_image):
            self.calc_mean()
        # mean = cv2.imread(mean_image)
        subtracted_img = images - 126
        return subtracted_img / 255.

    def calc_mean(self):
        pass



def define_model(input_shape, modelname='model.h5', gpu=-1):
    if gpu < 0:
        print("Using CPU and LeNet...")
        model = Sequential()
        # model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=X.shape[1:]))
        # model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')) #, dim_ordering='tf'))
        model.add(BatchNormalization())
        model.add(Conv2D(64, (3, 3), activation='relu'))
        model.add(MaxPooling2D(pool_size=(2, 2), strides=None, border_mode='valid')) #, dim_ordering='tf'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        model.add(Conv2D(32, (3, 3), activation='relu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(1))
    else:
        print("Using GPU and VGG16...")

        # input_tensor = Input(shape=X.shape[1:])
        # in_model = Sequential()
        # in_model.add(Lambda(lambda x: x / 255. - 0.5, input_shape=X.shape[1:]))
        # in_model.add(Cropping2D(cropping=((70, 25), (0, 0))))
        # vgg16_model = VGG16(include_top=False, weights='imagenet', input_tensor=in_model.output)

        vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=input_shape)
        fc_model = Sequential()
        fc_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
        fc_model.add(Dense(512, activation=keras.layers.advanced_activations.ELU(alpha=1.0),
                           kernel_initializer='he_normal'))
        fc_model.add(BatchNormalization())
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(128, activation=keras.layers.advanced_activations.ELU(alpha=1.0),
                           kernel_initializer='he_normal'))
        fc_model.add(BatchNormalization())
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(32, activation='relu', kernel_initializer='he_normal'))
        fc_model.add(Dropout(0.5))
        fc_model.add(Dense(1, name='output', kernel_initializer='he_normal'))
        
        model = Model(input=vgg16_model.input, output=fc_model(vgg16_model.output))
        
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    return model


def training(BatchLoader, model, epochs, iter_per_epoch=None, modelname='model.h5'):
    BatchLoader.train = True
    BatchLoaderTest = copy.copy(BatchLoader)
    BatchLoaderTest.train = False
    # if iter_per_epoch is None:
    #     ipe_train = BatchLoader.datasize_train / BatchLoader.batchsize
    #     ipe_test = BatchLoader.datasize_test / BatchLoader.batchsize
    # else:
    #     ipe_train = iter_per_epoch
    #     ipe_test = iter_per_epoch
    # model.fit_generator(BatchLoader, steps_per_epoch=ipe_train, epochs=epochs, 
    #                     validation_data=BatchLoaderTest, validation_steps=ipe_test)

    model.fit_generator(BatchLoader, steps_per_epoch=BatchLoader.datasize_train / BatchLoader.batchsize, epochs=epochs, 
                        validation_data=BatchLoaderTest, validation_steps=BatchLoader.datasize_test / BatchLoader.batchsize)

    model.save(modelname)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--gpu', '-g', type=int, default=-1,
                        help='GPU flag')
    parser.add_argument('--ipe', '-i', default=None,
                        help='Iteration per epoch (None = all)')
    parser.add_argument('--epochs', '-e', type=int, default=10,
                        help='Number of epochs')
    args = parser.parse_args()

    m = MiniBatchLoader(DATA_DIR, 32)
    model = define_model((80, 320, 3), gpu=args.gpu)
    training(m, model, epochs=args.epochs, iter_per_epoch=args.ipe, modelname='model.h5')



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