#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:10:25 2019

@author: quibim
"""

import pickle
from pathlib import Path
import numpy as np

def classify(textures):
    model = pickle.load(open(Path('classifiers',
                                  'logisticRegression_classifier.sav'), 'rb'))
    mean = np.load(Path('classifiers', 
                        'mean_textures.npy'))
    std = np.load(Path('classifiers', 
                        'std_textures.npy'))

    textures = textures - mean
    textures = textures / std

    label = model.predict(textures)
    
    return label