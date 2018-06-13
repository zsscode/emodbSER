# -*- coding: utf-8 -*-
"""
Created on Wed May 11 10:04:35 2016

@author: Luv
"""

#!/usr/bin/env python
#coding: utf-8
""" This work is licensed under a Creative Commons Attribution 3.0 Unported License.
    Frank Zalkow, 2012-2013 """

"""
    The original code is written for Python2, modified by krumo to support Python3
"""

import numpy as np
from matplotlib import pyplot as plt
import scipy.io.wavfile as wav
from numpy.lib import stride_tricks
from random import shuffle
import scipy
import os
import os.path
import _pickle as cPickle
import cv2

""" short time fourier transform of audio signal """


def stft(sig, frameSize, overlapFac=0.5, window=np.hanning):
    win = window(frameSize)
    hopSize = int(frameSize - np.floor(overlapFac * frameSize))

    # zeros at beginning (thus center of 1st window should be for sample nr. 0)
    # print(np.floor(frameSize/2.0))
    # print(sig)
    samples = np.append(np.zeros(np.floor(frameSize/2.0).astype(int)), sig)
    # cols for windowing
    cols = int(np.ceil((len(samples) - frameSize) / float(hopSize))) + 1
    # zeros at end (thus samples can be fully covered by frames)
    samples = np.append(samples, np.zeros(frameSize))

    frames = stride_tricks.as_strided(samples, shape=(cols, frameSize), strides=(
        samples.strides[0]*hopSize, samples.strides[0])).copy()
    frames *= win

    return np.fft.rfft(frames)


""" scale frequency axis logarithmically """


def logscale_spec(spec, sr=44100, factor=20.):
    timebins, freqbins = np.shape(spec)

    scale = np.linspace(0, 1, freqbins) ** factor
    scale *= (freqbins-1)/max(scale)
    scale = np.unique(np.round(scale)).astype(int)

    # create spectrogram with new freq bins
    newspec = np.complex128(np.zeros([timebins, len(scale)]))
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            newspec[:, i] = np.sum(spec[:, scale[i]:], axis=1)
        else:
            newspec[:, i] = np.sum(spec[:, scale[i]:scale[i+1]], axis=1)

    # list center freq of bins
    allfreqs = np.abs(np.fft.fftfreq(freqbins*2, 1./sr)[:freqbins+1])
    freqs = []
    for i in range(0, len(scale)):
        if i == len(scale)-1:
            freqs += [np.mean(allfreqs[scale[i]:])]
        else:
            freqs += [np.mean(allfreqs[scale[i]:scale[i+1]])]

    return newspec, freqs


""" plot spectrogram"""


def plotstft(audiopath, binsize=2**10, plotpath="/Users/Luv/Desktop/pics", colormap="jet"):
    samplerate, samples = wav.read(audiopath)
    s = stft(samples, binsize)

    sshow, freq = logscale_spec(s, factor=1.0, sr=samplerate)
    ims = 20.*np.log10(np.abs(sshow)/10e-6)  # amplitude to decibel
    ims = np.transpose(ims)
    print(ims.shape)
    return ims


"""read emodb into .pkl file 
   created by krumo"""


def read_emodb(img_rows=512, img_cols=128):
    rootdir = "./emodb/wav/"
    targetdir = "./emodb/spec/"
    colormap = "jet"
    num = 535

    data = np.empty((num, img_cols*img_rows))
    label = np.empty(num, dtype=int)
    i = 0
    label_W = []  # anger
    label_L = []  # boredom
    label_E = []  # disgust
    label_A = []  # fear/anxiety
    label_F = []  # happiness
    label_T = []  # sadness
    label_N = []  # neural
    for filename in os.listdir(rootdir):
        name = "".join(filename)
        if(name.endswith('wav') == True):
            full_name = rootdir+name
            ims = plotstft(full_name)
            imgs = scipy.misc.imresize(np.uint8(ims), (img_rows, img_cols))

            if filename[5] == 'W':  # anger
                name2 = targetdir+'anger/'+name[:7] + ".jpg"
                label[i] = 1

                label_W.append(i)
            elif filename[5] == 'L':  # boredom
                name2 = targetdir+'boredom/'+name[:7] + ".jpg"
                label[i] = 2
                label_L.append(i)
            elif filename[5] == 'E':  # disgust
                name2 = targetdir+'disgust/'+name[:7] + ".jpg"
                label[i] = 3
                label_E.append(i)
            elif filename[5] == 'A':  # anxiety/fear
                name2 = targetdir+'fear/'+name[:7] + ".jpg"
                label[i] = 4
                label_A.append(i)
            elif filename[5] == 'F':  # happiness
                name2 = targetdir+'happiness/'+name[:7] + ".jpg"
                label[i] = 5
                label_F.append(i)
            elif filename[5] == 'T':  # sadness
                name2 = targetdir+'sadness/'+name[:7] + ".jpg"
                label[i] = 6
                label_T.append(i)
            elif filename[5] == 'N':  # neutral
                name2 = targetdir+'neutral/'+name[:7] + ".jpg"
                label[i] = 0
                label_N.append(i)
            else:
                label[i] = 0
                print("error:"+str(i))
                label_N.append(i)
            cv2.imwrite(name2, imgs,)
            i = i+1
    f = open('./emodb.pkl', 'wb')
    cPickle.dump((data, label, label_W, label_L, label_E,
                  label_A, label_F, label_T, label_N), f)
    f.close()


read_emodb()
