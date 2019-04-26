#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 15:44:51 2019

@author: quibim
"""

import sys, getopt
import os
import numpy as np
import video_frames as vf
import view_classification as vc
import doppler_segmentation as ds
import texture_analysis as tex
import textures_classification as tc

import warnings
warnings.filterwarnings("ignore")

if __name__ == '__main__':
        
    try:
       opts, args = getopt.getopt(sys.argv[1:],"hf:",["help=","folder="])
       
    except getopt.GetoptError:
       print('main.py -f <videosfolder>')
       sys.exit(2)
       
    for opt, arg in opts:
       if opt == '-h':
          print('main.py -f <videosfolder>')
          sys.exit()
       elif opt in ("-f", "--folder"):
          videos_path = arg
         
#    videos_path = sys.argv[1]
    videos = os.listdir(videos_path)

    for v in videos:
        if not v.startswith('.'):
            file_path = os.path.join(videos_path, v)
            
            # 1. If doppler, extract frames from video
            if vf.if_doppler(file_path):
                frames = vf.load_video(file_path)
                
            else:
                continue
                
            segmentations = []
            anatomics = []

            for fr in frames:

                # 2. Segment colors by frame
                segmentedImage, anatomicImage = ds.segmentation(fr)
                
                segmentations.append(segmentedImage)
                anatomics.append(anatomicImage)
            
            # 3. Classify view. If long axis, extract texture fetaures
            if not vc.if_long_axis(anatomics):
                continue
            
            else:
                allTextures = []
                for s in segmentations:
                    
                    # 4. Texture analysis
                    if np.max(s) == 0:
                        continue
                    else:
                        textures = tex.textures(s)
                        allTextures.append(textures)
                    
            # Calculate mean and median of the textures features plus max velocity of the sequence
            numberOfFeatures = len(allTextures[0])
            
            textureFeatures = np.zeros((1, numberOfFeatures*2 + 1))
            textureFeatures[0,:numberOfFeatures] = np.nanmean(np.array(allTextures), axis=0)
            textureFeatures[0,numberOfFeatures:-1] = np.nanmedian(np.array(allTextures), axis=0)
            textureFeatures[0,-1] = max(allTextures)[-2]
            
            # 5. Supervised classifier
            label = tc.classify(textureFeatures)[0] 
            
            if label == 1:
                print('RHD')
            elif label == 0:
                print('Normal')
            else:
                print('Unspecific label')
        
        