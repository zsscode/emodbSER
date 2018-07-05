# -*- coding:utf-8 -*-
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten, Dropout, Activation, Flatten
from keras.models import Model, Sequential
from keras.datasets import mnist
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.utils import np_utils
from PIL import Image
from random import shuffle
import os
import _pickle as cPickle
from read_data import read_emodb

np.random.seed(1337)  # for reproducibililty

def load_data():
    f = open('./emodb.pkl', 'rb')
    data, label, label_W, label_L, label_E, label_A, label_F, label_T, label_N = cPickle.load(
        f)

    feature_length = data.shape[1]
    X_train = np.empty((464, feature_length))
    Y_train = np.empty(464, dtype=int)

    X_valid = np.empty((535, feature_length))
    Y_valid = np.empty(535, dtype=int)

    X_test = np.empty((71, feature_length))
    Y_test = np.empty(71, dtype=int)

    for i in range(535):
        if i < 464:
            X_train[i, :] = data[i, :]
            Y_train[i] = label[i]
        else:
            X_test[i-464, :] = data[i, :]
            Y_test[i-464] = label[i]
        X_valid[i, :] = data[i, :]
        Y_valid[i] = label[i]

    return X_train, Y_train, X_valid, Y_valid, X_test, Y_test


if __name__ == '__main__':
    batch_size = 10
    nb_classes = 7
    nb_epoch = 1

    # input image dimensions
    img_rows, img_cols = 512, 128
    # number of convolutional filters to use
    nb_filters = 32
    # size of pooling area for max pooling
    nb_pool = 2
    # convolution kernel size
    nb_conv = 3

    # read_emodb(img_rows, img_cols)

    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data()
    print(x_train.shape)

    x_train = x_train.astype('float32') / 255.0
    X_train = np.reshape(x_train, (len(x_train), 1, img_rows, img_cols))

    x_test = x_test.astype('float32') / 255.0
    X_test = np.reshape(x_test, (len(x_test), 1, img_rows, img_cols))

    x_valid = x_valid.astype('float32') / 255.0
    X_valid = np.reshape(x_valid, (len(x_valid), 1, img_rows, img_cols))

    # convert label to binary class matrix
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_valid = np_utils.to_categorical(y_valid, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)

    print('X_train shape:', X_train.shape)
    print(X_train.shape[0], 'train samples')
    print(X_test.shape[0], 'test samples')

    model = Sequential()

    # 32 convolution filters , the size of convolution kernel is 3 * 3
    # border_mode can be 'valid' or 'full'
    # ‘valid’only apply filter to complete patches of the image.
    # 'full'  zero-pads image to multiple of filter shape to generate output of shape: image_shape + filter_shape - 1
    # when used as the first layer, you should specify the shape of inputs
    # the first number means the channel of an input image, 1 stands for grayscale imgs, 3 for RGB imgs
    # dim_ordering="th" means "NCHW", dim_ordeing="tf" means "NHWC"
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv,
                            border_mode='same',
                            input_shape=(1, img_rows, img_cols), dim_ordering="th"))
    # use rectifier linear units : max(0.0, x)
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering="th"))
    # second convolution layer with 32 filters of size 3*3
    model.add(Convolution2D(nb_filters, nb_conv, nb_conv, dim_ordering="th"))
    model.add(Activation('relu'))
    # max pooling layer, pool size is 2 * 2
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool), dim_ordering="th"))
    # drop out of max-pooling layer , drop out rate is 0.25
    model.add(Dropout(0.25))
    # flatten inputs from 2d to 1d
    model.add(Flatten())
    # add fully connected layer with 128 hidden units
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    # output layer with softmax
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    # use cross-entropy cost and adadelta to optimize params

    model.compile(loss='categorical_crossentropy',
                  optimizer='adadelta',
                  metrics=['accuracy'])

    model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch,
              verbose=1, validation_data=(X_valid, Y_valid))
    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
