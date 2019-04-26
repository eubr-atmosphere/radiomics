#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  1 09:23:45 2019

@author: anajimenezpastor
"""

import numpy as np
import os
import cv2

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

def test():
    work_path = os.getcwd()
    
    # Loading image
    img = cv2.imread(os.path.join(work_path, 'test.jpg'))
    
    # Define CNN model and load weights
    model = classifier()
    model.load_weights(os.path.join(work_path, 'classifier.hdf5'))
    
    # RGB image to grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Image normalization          
    mean = np.load(os.path.join(work_path, 'mean.npy'))
    std = np.load(os.path.join(work_path, 'std.npy'))     
    img = samplewise_intensity_normalization(img)
    img = featurewise_normalization(img, mean, std)
    
    # Label prediction
    img = img[..., np.newaxis]
    img = img[np.newaxis, ...]
    pred = model.predict(img)
    pred = int(np.argmax(pred))
    
    if pred==0: 
        print('Four Chamber') 
    elif pred == 1: 
        print('Short Axis') 
    elif pred == 2: 
        print('Long Axis')
        
if __name__ == '__main__': 
    test() 
    
    