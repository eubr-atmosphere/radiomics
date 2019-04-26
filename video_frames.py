# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 11:09:54 2018

@author: quibim
"""

import cv2
import numpy as np

def load_video(path):
    
    cap = cv2.VideoCapture(path)
    
    frames = []
    while(True):
        # Capture frame-by-frame
        ret, frame = cap.read()
        if ret == True:
            frames.append(frame)
        else:
            break
    
    # When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return np.array(frames)


def classify_video(frames):
    
    frame = frames[0]
    
    # Convert BGR to HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # define range of blue color in HSV
    lower_blue = np.array([110,50,50])
    upper_blue = np.array([130,255,255])
    
    # define range of red color in HSV
    lower_red = np.array([169,50,50])
    upper_red = np.array([189,255,255])

    # Threshold the HSV image to get only blue colors
    mask_blue = cv2.inRange(hsv, lower_blue, upper_blue)
    mask_red = cv2.inRange(hsv, lower_red, upper_red)
    # Threshold obtenido empíricamente
    if mask_blue.sum() < 25000 and mask_red.sum() < 25000:
        video_class = 'anatomic'
    else:
        video_class = 'doppler'
    
    return video_class

        
def if_doppler(file_path):
    
    frames = load_video(file_path)
    video_class = classify_video(frames)
    
    if video_class == 'doppler':
        return True
    else:
        return False
    
