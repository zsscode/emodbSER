# -*- coding:utf-8 -*-
from keras.layers import Input, Dense, Convolution2D, MaxPooling2D, UpSampling2D, Flatten
from keras.models import Model
from keras.datasets import mnist
import numpy as np
# use Matplotlib (don't ask)
import matplotlib.pyplot as plt
from keras.utils import np_utils
from random import shuffle
from PIL import Image
import _pickle as cPickle
import os

img_rows = 512
img_cols = 128
batches = 10
auto_epoch = 1
classify_epoch = 1


def load_data(file):
    f = open(file, 'rb')
    data, label, label_W, label_L, label_E, label_A, label_F, label_T, label_N = cPickle.load(
        f)
    f.close()
    print(data)
    X_train = np.empty((464, img_cols*img_rows))
    Y_train = np.empty(464, dtype=int)

    X_valid = np.empty((len(data), img_cols*img_rows))
    Y_valid = np.empty(len(data), dtype=int)

    X_test = np.empty((71, img_cols*img_rows))
    Y_test = np.empty(71, dtype=int)

    for i in range(len(data)):
        if i < 464:
            X_train[i, :] = data[i, :]
            Y_train[i] = label[i]
        else:
            X_test[i-464, :] = data[i, :]
            Y_test[i-464] = label[i]
        X_valid[i, :] = data[i, :]
        Y_valid[i] = label[i]

    return (X_train, Y_train, X_valid, Y_valid, X_test, Y_test)


def load_pretrain(file):
    f = open(file, 'rb')
    data, label, label_W, label_L, label_E, label_A, label_F, label_T, label_N = cPickle.load(
        f)
    f.close()

    X_train = np.empty((len(data), img_cols*img_rows))
    Y_train = np.empty(len(data), dtype=int)

    for i in range(len(data)):
        X_train[i, :] = data[i, :]
        Y_train[i] = label[i]
    return X_train, Y_train

def run_model():
    x_train, y_train, x_valid, y_valid, x_test, y_test = load_data(
        './emodb.pkl')

    x_train = x_train.astype('float32') / 255.0
    x_train = np.reshape(x_train, (len(x_train), 1, img_rows, img_cols))

    x_test = x_test.astype('float32') / 255.0
    x_test = np.reshape(x_test, (len(x_test), 1, img_rows, img_cols))

    x_valid = x_valid.astype('float32') / 255.0
    x_valid = np.reshape(x_valid, (len(x_valid), 1, img_rows, img_cols))

    Y_train = np_utils.to_categorical(y_train, 7)
    Y_valid = np_utils.to_categorical(y_valid, 7)
    Y_test = np_utils.to_categorical(y_test, 7)

    print('X_train shape:', x_train.shape)
    print(x_train.shape[0], 'train samples')
    print(x_test.shape[0], 'test samples')

    input_img = Input(shape=(1, img_rows, img_cols))

    x = Convolution2D(16, 3, 3, activation='relu',
                      border_mode='same', dim_ordering="th")(input_img)
    x = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)
    x = Convolution2D(8, 3, 3, activation='relu',
                      border_mode='same', dim_ordering="th")(x)
    encoded = MaxPooling2D((2, 2), border_mode='same', dim_ordering="th")(x)

    # at this point the representation is (8, 4, 4) i.e. 128-dimensional
    x = Convolution2D(8, 3, 3, activation='relu',
                      border_mode='same', dim_ordering="th")(encoded)
    x = UpSampling2D((2, 2), dim_ordering="th")(x)
    x = Convolution2D(16, 3, 3, activation='relu',
                      border_mode='same', dim_ordering="th")(x)
    x = UpSampling2D((2, 2), dim_ordering="th")(x)
    decoded = Convolution2D(1, 3, 3, activation='sigmoid',
                            border_mode='same', dim_ordering="th")(x)

    # classifier

    y = Flatten()(encoded)
    y = Dense(128, activation='relu')(y)
    classifier = Dense(7, activation='softmax')(y)

    autoencoder = Model(input_img, decoded)
    autoencoder.compile(optimizer='adadelta', loss='binary_crossentropy')
    autoencoder.fit(x_train, x_train,
                    nb_epoch=auto_epoch,
                    batch_size=batches,
                    shuffle=True,
                    validation_data=(x_test, x_test)
                    )

    decoded_imgs = autoencoder.predict(x_test)

    # show some reconstruction samples
    """
    n = 10
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        print(str(i)+" pic")
        ax = plt.subplot(2, n, i+1)
        plt.imshow(x_test[i].reshape(img_rows,img_cols))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + n+1)
        plt.imshow(decoded_imgs[i].reshape(img_rows,img_cols))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
    plt.show()
    """

    classification = Model(input_img, output=classifier)
    classification.compile(optimizer='rmsprop',
                           loss='categorical_crossentropy',
                           metrics=['accuracy'])
    classification.fit(x_train, Y_train,
                       nb_epoch=classify_epoch,
                       batch_size=batches,
                       verbose=1,
                       shuffle=True,
                       validation_data=(x_valid, Y_valid))

    score = classification.evaluate(x_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])

if __name__== "__main__":
    run_model()