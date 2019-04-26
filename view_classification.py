#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:51:52 2019

@author: quibim
"""

import numpy as np
import cv2
from pathlib import Path
from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input, Conv2D, MaxPooling2D
from keras.optimizers import Adam
from keras import backend as K
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

image_rows = 320
image_cols = 240
n_labels = 3

def featurewise_normalization(img, mean, std):
    img = (img - mean)/std
    return img

def samplewise_intensity_normalization(img):
    maxim = np.max(img)
    minim = np.min(img)
    img = (img - minim) / (maxim - minim)
    return img

def classifier(img_rows = image_rows, img_cols = image_cols, img_channels = 1, n_labels = n_labels):
    
    inputs = Input((img_rows,img_cols, img_channels))
    
    conv1 = Conv2D(32, (3, 3), padding='same')(inputs)
    conv1 = Activation('relu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv1)
    
    conv2 = Conv2D(64, (3, 3), padding='same')(pool1)
    conv2 = Activation('relu')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv2)
    
    conv3 = Conv2D(128, (3, 3), padding='same')(pool2)
    conv3 = Activation('relu')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv3)
    
    conv4 = Conv2D(256, (3, 3), padding='same')(pool3)
    conv4 = Activation('relu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv4)
    
    conv5 = Conv2D(512, (3, 3), padding='same')(pool4)
    conv5 = Activation('relu')(conv5)
    pool5 = MaxPooling2D(pool_size=(2, 2), strides = (2,2))(conv5)
    
    output = Flatten()(pool5)
    output = Dense(n_labels)(output)
    output = Activation('softmax')(output)
    
    model = Model(inputs=[inputs], outputs=[output])
     
    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-3), metrics=['accuracy'])
    
    return model

def view_classification(frames):
    
    # Define CNN model and load weights
    model = classifier()
    model.load_weights(Path('classifiers', 'classifier.hdf5'))
    
    mean = np.load(Path('classifiers', 'mean.npy'))
    std = np.load(Path('classifiers', 'std.npy'))
    letters = np.load(Path('classifiers', 'letters.npy'))
    
    predictions = []
    
    for img in frames:
        
        h,w,_ = img.shape
        img[:,-8:,:] = [0,0,0]
        for i in range(len(letters)-4):
            let = letters[i,:]
            img = cv2.rectangle(img, (int(let[0])-15, h - int(let[1]) +3), (int(let[2])+3, h - int(let[3])-6), (0, 0, 0), -1)
        letV = letters[-1,:]
        img = cv2.rectangle(img, (int(letV[0])-2, h - int(letV[1]) +2), (int(letV[2])+3, h - int(letV[3])-2), (0, 0, 0), -1)
        letCM = np.array([199, h-295, 199+34,h-(300+20)])
        img = cv2.rectangle(img, (int(letCM[0])-2, h - int(letCM[1]) +2), (int(letCM[2])+3, h - int(letCM[3])-2), (0, 0, 0), -1)        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        img = samplewise_intensity_normalization(img)
        img = featurewise_normalization(img, mean, std)
        img = img[..., np.newaxis]
        img = img[np.newaxis, ...]
        pred = model.predict(img)
        predictions.append(np.argmax(pred))
        
    counts = np.bincount(predictions)
    
    # 0 --> 4 chamber
    # 1 --> Short axis
    # 2 --> Long axis
    
    return np.argmax(counts)

def if_long_axis(frames):
    
    view = view_classification(frames)
    
    if view == 2:
        return True
    else:
        return False
    
        