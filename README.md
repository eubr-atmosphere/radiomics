# Radiomics Approach

This application focuses on the implementation of the pilot application on Medical Imaging Biomarkers. This radiomics approach includes a processing pipeline to extract frames from videos, classify them, select those frames with significative data, filter them and extract image features using first- and second-order texture analysis and image computing. Finally, that pipeline concludes a classification (normal, definite or borderline RHD). 

## Functions

* main: Main pipeline
* video_frame: Read video frame and detect doppler ones
* view_classification: Classify video frames into the different view classes:
  * 0: 4 chamber
  * 1: Parasternal Short Axis
  * 2: Parasternal Long Axis
* doppler_segmentation: Apply a color-based segmentation to extract the colors from the doppler images using a k-means clustering
* texture_analysis: Perform first- and second-order texture analysis for image characterization and extraction of maximum blood velocities
* texture_classification: Conclude a classification (normal, borderline or definite RHD) according to the image features

 ## Folders
 
 * classifiers: includes files needed for the view classifier and for the textures classification


## Usage

main.py -f videosfolder 