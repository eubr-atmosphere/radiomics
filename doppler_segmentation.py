#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:12:41 2019

@author: quibim
"""

import numpy as np
import cv2
from skimage.color import rgb2lab

def segmentation(img):
    
    ##################### COLOR-BASED SEGMENTATION ###############################
    
    #Black boxes to remove colorbar and yellow lines
    nrows, ncols, dim = img.shape

    img[1:150, 1:25, :] = 0
    img[:, 230:ncols, :] = 0
    img[305:nrows, 200:ncols, :] = 0
    
    # Convert the image into a L*A*B colorspace
    labImage = rgb2lab(img)
    
    # First, it is needed to reshape the image to have an array of Mx3 size (due to 
    # the 3 features: R,G,B)
    Z = labImage.reshape((-1,3))
    
    # Inputs: image (float32), clusters number,bestlabels, criteria, attemps, flags
    Z = np.float32(Z)
    nclusters = 8
    bestlabels = None
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 0.01) 
    attempts = 10
    flags = cv2.KMEANS_PP_CENTERS #How the initial centers are taken
    
    # Apply K-Means
    ret,label,center=cv2.kmeans(Z,nclusters,bestlabels,criteria,attempts,flags)
    
    center = np.uint8(center)
    
    # Create and save the images
    dopplerImage = 0
    anatomicImage = 0
    
    for c in range(nclusters):
        seg = np.copy(img)
        seg = seg.reshape((-1,3))
        seg[label.flatten()!=c] = 0
        seg = seg.reshape(nrows,ncols,dim)
        
        if abs(center[c,0]) > 10 and ((center[c,1] > 20 and center[c,1] < 230) or (center[c,2] > 20 and center[c,2] < 230)):
            dopplerImage = dopplerImage + seg
        else:
            anatomicImage = anatomicImage + seg
    
    # Create mask to remove yellow box
    dopplerGray = cv2.cvtColor(dopplerImage, cv2.COLOR_BGR2GRAY)
    dopplerBin = cv2.threshold(dopplerGray, 1, 255, cv2.THRESH_BINARY)[1]
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,3))
    
    mask = cv2.erode(dopplerBin,kernel,iterations = 1)

    dopplerMasked = cv2.bitwise_and(dopplerImage,dopplerImage,mask = mask)
   
    return dopplerMasked, anatomicImage
