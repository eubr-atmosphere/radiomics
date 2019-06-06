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
import hashlib
import PIL.Image

import warnings
warnings.filterwarnings("ignore")

def extract_and_anonymise(videos_path,input_v,videos_path_out,output_v):
    vfull_in=os.path.join(videos_path, input_v)
    print('Extracting and anonymising: %s' % vfull_in)
    frames = vf.load_video(vfull_in)
    index = 0
    for fr in frames:
        dimy, dimx, depth = fr.shape
        for i in range(0,int(dimy/6)):
            for j in range(int(2*dimx/3)-10,dimx):
                fr[i][j][0]=0
                fr[i][j][1]=0
                fr[i][j][2]=0
                
        directory=os.path.join(videos_path_out, output_v)
        if not os.path.exists(directory):
            os.makedirs(directory)

        vfull_out=os.path.join(directory, str(index)+'.png')
        index = index+1
        imagen=PIL.Image.fromarray(fr)
        imagen.save(vfull_out, 'png')



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
         
    videos = os.listdir(videos_path)
    videos_path_out = videos_path + '-anonymised'
    for v in videos:
        if not v.startswith('.'):
            file_path=os.path.join(videos_path, v)
            print('Check %s' % file_path)
            if vf.if_doppler(file_path):
                hash_object = hashlib.md5(file_path.encode())
                output_v=hash_object.hexdigest()
                extract_and_anonymise(videos_path,v,videos_path_out,output_v)
