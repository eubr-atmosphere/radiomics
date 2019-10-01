#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 11:08:01 2019

@author: Ana Jimenez
"""

from keras.models import Model
from keras.layers import Dense, Activation, Flatten, Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.optimizers import Adadelta
from keras import backend as K
from keras.callbacks import ModelCheckpoint, Callback
K.set_image_data_format('channels_last')  # TF dimension ordering in this code

import numpy as np
import cv2
import os

import tensorflow as tf
import math
import horovod.keras as hvd


def featurewise_normalization(images):
    mean = np.mean(images)
    std = np.std(images)
#    np.save(os.path.join(paths['main_path'], 'training','mean.npy'),mean)
#    np.save(os.path.join(paths['main_path'], 'training','std.npy'),std)
    images = np.subtract(images,mean)
    images = np.divide(images,std)
    return images

def samplewise_intensity_normalization(images):
    for i in range(images.shape[0]):
        img = images[i,:,:]
        maxim = np.max(img)
        minim = np.min(img)
        images[i,:,:] = (img - minim) / (maxim - minim)
    return images

def augmentation_rotation(image,augmentation_params):
    image_rot = np.copy(image)
    
    if(flip(augmentation_params['rotation_prob']) == 'augment'):   
        rows,cols, channels = image.shape
        angle = np.random.uniform(augmentation_params['min_angle'], augmentation_params['max_angle'])
        M = cv2.getRotationMatrix2D((cols/2,rows/2),angle,1)
        image_rot = cv2.warpAffine(image,M,(cols,rows))[..., np.newaxis]

    return image_rot
  

def flip(p):
    return 'augment' if np.random.random() < p else 'pass'

def training_data_augmentation(img, augmentation_params):
        
    img_ = augmentation_rotation(img, augmentation_params)
    
    img_ = img_.astype('float32')
    
    return img_

def data_generator(X_train, y_train, augmentation_params, batch_size):
    
    batch_images = np.zeros([batch_size, X_train.shape[1], X_train.shape[2], X_train.shape[3]])
    batch_labels = np.zeros([batch_size,  y_train.shape[1]])
    index = 0
    while True:
        for b in range(batch_size):
            if index == X_train.shape[0]:
                index = 0
                shuffle_indexes = np.arange(X_train.shape[0])
                np.random.shuffle(shuffle_indexes)
                X_train = X_train[shuffle_indexes]
                y_train = y_train[shuffle_indexes]
                
            img = training_data_augmentation(X_train[index], augmentation_params)
            
            batch_images[b] = img
            batch_labels[b] = y_train[index]
            
            index += 1
        yield batch_images, batch_labels

def classifier():
    
    inputs = Input((image_params['img_rows'],image_params['img_cols'], image_params['img_channels']))
    
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
    output = Dense(image_params['n_labels'])(output)
    output = Activation('softmax')(output)
    
    model = Model(inputs=[inputs], outputs=[output])
         
    return model
    
def train():
                
    print('-'*30)
    print('Loading train data...')
    print('-'*30)
    
    x_train = np.load(os.path.join(paths['main_path'], 'data' , 'x_train.npy'))
    y_train = np.load(os.path.join(paths['main_path'], 'data', 'y_train.npy'))
    x_val = np.load(os.path.join(paths['main_path'], 'data', 'x_val.npy'))
    y_val = np.load(os.path.join(paths['main_path'], 'data', 'y_val.npy'))
    
    indexes = np.arange(x_train.shape[0])
    np.random.shuffle(indexes)
    
    x_train = x_train[indexes]
    y_train = y_train[indexes]
    
    indexes = np.arange(x_val.shape[0])
    np.random.shuffle(indexes)
    
    x_val = x_val[indexes]
    y_val = y_val[indexes]
    
    
    if image_params['samplewise_intensity_normalization'] == True:
        x_train = samplewise_intensity_normalization(x_train)
        x_val = samplewise_intensity_normalization(x_val)
    
    if image_params['featurewise_normalization'] == True:
        x_train = featurewise_normalization(x_train)
        x_val = featurewise_normalization(x_val)   
        
    x_train = x_train[..., np.newaxis]
    x_val = x_val[..., np.newaxis]
    
    if not os.path.isdir('training'):
        os.mkdir('training')
    
    print('-'*30)
    print('Creating and compiling model...')
    print('-'*30)
    
    # Horovod: initialize Horovod.
    hvd.init()
    
    #Horovod: pin GPU to be used to process local rank (one GPU per process)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.gpu_options.visible_device_list = str(hvd.local_rank())
    K.set_session(tf.Session(config=config))
    
    # Horovod: adjust number of epochs based on number of GPUs.
    epochs = int(math.ceil(12.0 / hvd.size()))
    
    # Horovod: adjust learning rate based on number of GPUs.
    opt = Adadelta(1.0 * hvd.size())
    
    # Horovod: add Horovod Distributed Optimizer.
    opt = hvd.DistributedOptimizer(opt)
          
    data_generator_train = data_generator(x_train, y_train, augmentation_params, training_params['batch_size'])
    
    print('-'*30)
    print('Fitting model...')
    print('-'*30)
    model = classifier()
    model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
    
    callbacks = [
    # Horovod: broadcast initial variable states from rank 0 to all other processes.
    # This is necessary to ensure consistent initialization of all workers when
    # training is started with random weights or restored from a checkpoint.
    hvd.callbacks.BroadcastGlobalVariablesCallback(0),
    ]   
    
    # Horovod: save checkpoints only on worker 0 to prevent other workers from corrupting them.
    if hvd.rank() == 0:
        callbacks.append(ModelCheckpoint(filepath=(os.path.join(paths['main_path'], 'training', 'classifier.hdf5')), monitor='val_loss', save_best_only=True))


    model.fit_generator(generator = data_generator_train, 
                        steps_per_epoch = x_train.shape[0]//training_params['batch_size'],
                        epochs = epochs,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data = (x_val, y_val))



if __name__ == '__main__':
    
    paths = {
            'main_path' : '/mnt/share/vol001/views_classification'
            }

    image_params = {
                    'img_rows': 320, 
                    'img_cols': 240,
                    'img_channels': 1,
                    'n_labels': 3,
                    'featurewise_normalization': True,
                    'samplewise_intensity_normalization': True
                    }
    
    augmentation_params = {
                            'rotation': True,
                            'min_noise_var':0.001,
                            'max_noise_var':0.01,
                            'min_angle': -5,
                            'max_angle': 5,
                            'rotation_prob': 0.3
                        }
    
    training_params = {
                        'val_fraction':0.2,
                        'architecture': 'classifier',
                        'batch_size': 40,
                        }
    
    train()
