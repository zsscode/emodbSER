# emodbSER

## Description

This project aims to realize Speech Emotion Recognition task on speech dataset of [Berlin Database of Emotional Speech](http://emodb.bilderbar.info/docu/) using deep convolutional neural network. With the help of autoencoder technique to pretrain the network, compared to the Convolutional Neural Network without pretraining our structure could improve the recognition accuracy from 20% to 60%. The Neural Networks are implemented based on [Keras](https://keras.io/) framework.

## Prerequisites
* Python3
* Keras
* Any backend supported by Keras like [Tensorflow](https://github.com/tensorflow/tensorflow), [MXNet](https://github.com/apache/incubator-mxnet)

## Usage
1. Download Berlin Database of Emotional Speech from [here](http://www.emodb.bilderbar.info/download/) and rename the dataset folder as emodb.
2. Run `read_data.py` to generate `emodb.pkl`
3. Modify the neural network architecture and hyperparameters in `keras_cnn.py` and `keras_cae.py`.
4. Train and test the model by running commands `python keras_cnn.py` and `python keras_cae.py`
