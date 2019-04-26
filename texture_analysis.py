#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 11 10:06:50 2019

@author: quibim
"""

import numpy as np
import cv2
from scipy.stats import kurtosis, skew
import texture

listOfFeatures = np.array(['autoCorrelation','clusterProminence','clusterShade','contrast','correlation','differenceEntropy',
                           'differenceVariance','dissimilarity','energy','entropy','homogeneity',
                           'informationMeasureOfCorrelation1','informationMeasureOfCorrelation2','inverseDifference',
                           'maximumProbability','sumAverage','sumEntropy','sumOfSquaresVariance','sumVariance', 'kurtosis',
                           'skewness','mean','std','median','p25','p75','maxVelocity','meanVelocity'])


def GLCM_features(glcm):
    
    # Get size of glcm
    nGrayLevels = glcm.shape[0]
    nglcm = glcm.shape[3]
    
    class Radiomics:
        autoCorrelation                     = np.zeros((1,nglcm)) # Autocorrelation
        clusterProminence                   = np.zeros((1,nglcm)) # Cluster Prominence
        clusterShade                        = np.zeros((1,nglcm)) # Cluster Shade
        contrast                            = np.zeros((1,nglcm)) # Contrast
        correlation                         = np.zeros((1,nglcm)) # Correlation
        differenceEntropy                   = np.zeros((1,nglcm)) # Difference entropy
        differenceVariance                  = np.zeros((1,nglcm)) # Difference variance
        dissimilarity                       = np.zeros((1,nglcm)) # Dissimilarity
        energy                              = np.zeros((1,nglcm)) # Energy
        angularSecondMoment                 = np.zeros((1,nglcm)) # Angular second moment
        entropy                             = np.zeros((1,nglcm)) # Entropy
        homogeneity                         = np.zeros((1,nglcm)) # Homogeneity
        informationMeasureOfCorrelation1    = np.zeros((1,nglcm)) # Information measure of correlation1
        informationMeasureOfCorrelation2    = np.zeros((1,nglcm)) # Informaiton measure of correlation2
        inverseDifference                   = np.zeros((1,nglcm)) # Homogeneity
        maximumProbability                  = np.zeros((1,nglcm)) # Maximum probability
        sumAverage                          = np.zeros((1,nglcm)) # Sum average
        sumEntropy                          = np.zeros((1,nglcm)) # Sum entropy
        sumOfSquaresVariance                = np.zeros((1,nglcm)) # Sum of sqaures: Variance
        sumVariance                         = np.zeros((1,nglcm)) # Sum variance

    glcmMean    = np.zeros((nglcm,1))
    uX          = np.zeros((nglcm,1))
    uY          = np.zeros((nglcm,1))
    sX          = np.zeros((nglcm,1))
    sY          = np.zeros((nglcm,1))
    
    # pX pY pXplusY pXminusY
    pX          = np.zeros((nGrayLevels,nglcm))
    pY          = np.zeros((nGrayLevels,nglcm))
    pXplusY     = np.zeros(((nGrayLevels*2 - 1),nglcm))
    pXminusY    = np.zeros(((nGrayLevels),nglcm))
    
    # HXY1 HXY2 HX HY
    HXY1        = np.zeros((nglcm,1))
    HX          = np.zeros((nglcm,1))
    HY          = np.zeros((nglcm,1))
    HXY2        = np.zeros((nglcm,1))
    
    # Create indices for vectorising code:
    sub   = range(nGrayLevels*nGrayLevels)
    I, J  = np.unravel_index(sub, [nGrayLevels,nGrayLevels], order='F')

    I = I + 1
    J = J + 1
    
    radiomics = Radiomics()
    
    radiomics.contrast                      = texture.greycoprops(glcm, 'contrast')
    radiomics.dissimilarity                 = texture.greycoprops(glcm, 'dissimilarity')
    radiomics.homogeneity                   = texture.greycoprops(glcm, 'homogeneity')
    radiomics.correlation                   = texture.greycoprops(glcm, 'correlation')
    
    for k in range(nglcm):
        
        currentGLCM = glcm[:,:,0,k]
        glcmMean[k] = np.mean(currentGLCM)
        
        index = np.unravel_index(sub, currentGLCM.shape, 'F')
        
        uX[k]   = np.sum(I*currentGLCM[index])
        uY[k]   = np.sum(J*currentGLCM[index])
        sX[k]   = np.sum((I-uX[k])**2*currentGLCM[index])
        sY[k]   = np.sum((J-uY[k])**2*currentGLCM[index])
            
        radiomics.energy[0,k]               = np.sum(currentGLCM[index]**2)
        radiomics.inverseDifference[0,k]    = np.sum(currentGLCM[index]/(1 + abs(I-J)))
        radiomics.sumOfSquaresVariance[0,k] = np.sum(currentGLCM[index]*((I - uX[k])**2))
        radiomics.maximumProbability[0,k]   = np.max(currentGLCM[:])
        
        pX[:,k]   = np.sum(currentGLCM,1)
        pY[:,k]   = np.sum(currentGLCM,0)
        
        tmp1 = np.array([(I+J),currentGLCM[index]])
        tmp2 = np.array([abs((I-J)), currentGLCM[index]])
        idx1 = np.arange(2, 2*nGrayLevels + 1)
        idx2 = np.arange(nGrayLevels)
        
        for i in idx1:
            pXplusY[i-2,k] = np.sum(tmp1[1, tmp1[0,:]==i])

        for i in idx2:
            pXminusY[i,k] = np.sum(tmp2[1, tmp2[0,:]==i])
        
        radiomics.sumAverage[0,k]                = np.sum((idx1)*(pXplusY[:,k]))
        radiomics.sumEntropy[0,k]                = -np.nansum(pXplusY[:,k]*np.log(pXplusY[:,k]))
        radiomics.differenceEntropy[0,k]         = -np.nansum(pXminusY[:,k]*np.log(pXminusY[:,k]))
        radiomics.differenceVariance[0,k]   = np.sum(((idx2-radiomics.dissimilarity[0,k])**2).transpose()*pXminusY[idx2,k])
        radiomics.sumVariance[0,k]          = np.sum((idx1-radiomics.sumAverage[0,k]).transpose()**2*pXplusY[idx1-2,k])

        HXY1[k]                     = -np.nansum(currentGLCM[index].transpose()*np.log(pX[I-1,k]*pY[J-1,k]))
        HXY2[k]                     = -np.nansum(pX[I-1,k]*pY[J-1,k]*np.log(pX[I-1,k]*pY[J-1,k]))
        HX[k]                       = -np.nansum(pX[:,k]*np.log(pX[:,k]))
        HY[k]                       = -np.nansum(pY[:,k]*np.log(pY[:,k]))
        
        radiomics.autoCorrelation[0,k]       = np.sum(I*J*currentGLCM[index])
        radiomics.clusterProminence[0,k]     = np.sum((I+J-uX[k]-uY[k])**4*currentGLCM[index])
        radiomics.clusterShade[0,k]          = np.sum((I+J-uX[k]-uY[k])**3*currentGLCM[index])
        radiomics.entropy[0,k]               = -np.nansum(currentGLCM[index]*np.log(currentGLCM[index]))
        radiomics.inverseDifference[0,k]     = np.sum(currentGLCM[index]/( 1 + abs(I-J) ))

        radiomics.informationMeasureOfCorrelation1[0,k]  = (radiomics.entropy[0,k]-HXY1[k])/(max(HX[k],HY[k]))
        radiomics.informationMeasureOfCorrelation2[0,k]  = (1 - np.exp(-2*(HXY2[k]-radiomics.entropy[0,k])))**(1/2)
        
    return radiomics
      

def textures(img):
    
    # Convert to gray
    Z = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Convert to 64 gray levels
    Z = np.float16(np.round(Z/4))
    
    # Put as nans those non-color pixels
    Z[Z == 0] = np.nan
    
    # Inputs:image, distances, angles, levels, symmetric, normed
    distances = [1]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    grayLevels = 64
    
    if not np.nanmax(Z) == 0:
        glcm = texture.greycomatrix_with_nan(Z, distances, angles, grayLevels, symmetric=False, normed=True)
        radiomics = GLCM_features(glcm)
        
    class radiomicsConc():
        pass
    
    for l in listOfFeatures:
        try:
            setattr(radiomicsConc, l, np.nanmean(getattr(radiomics,l)))
        except:
            break
            
    # The background must not be nan for gaussian blur and for the kurtosis/skewness stats 
    Z = np.uint8(Z) 

    ind = np.nonzero(Z)
    voxels = Z[ind]
    
    radiomicsConc.kurtosis      = kurtosis(voxels.flatten(), axis=0, fisher=False)
    radiomicsConc.skewness      = skew(voxels.flatten())
    radiomicsConc.mean          = np.mean(voxels.flatten())
    radiomicsConc.std           = np.std(voxels.flatten())
    radiomicsConc.median        = np.median(voxels.flatten())
    radiomicsConc.p25           = np.percentile(voxels.flatten(),25)
    radiomicsConc.p75           = np.percentile(voxels.flatten(),75)

    # Calculation of maximum and mean velocity in the image
    maxDopplerVel = 0.64 #Define maximum doppler velocity
    
    # Smooth the image to avoid noisy maximums
    fwhm = 8
    sigma = fwhm / np.sqrt(8 * np.log(2))
    
    Zblur = cv2.GaussianBlur(Z, (3,3), sigma)
    
    _, maxVal, _, _ = cv2.minMaxLoc(Zblur)
    
    # Max velocity
    radiomicsConc.maxVelocity = maxDopplerVel*maxVal/grayLevels
    
    # Mean velocity
    radiomicsConc.meanVelocity = maxDopplerVel*radiomicsConc.mean/grayLevels
    
    features = []
    for l in listOfFeatures:
        features.append(getattr(radiomicsConc,l))
                
    return features