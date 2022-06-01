# -*- coding: utf-8 -*-
"""
Created on Fri May 13 23:15:30 2022

@author: vijay
"""
import tensorflow as tf

import matplotlib.pyplot as plt
import cv2 
import os
import random
import numpy as np
import glob
import ntpath



def reshaping(data, img_ht, img_wt):
    X_train_flattened = data.reshape(len(data), img_ht, img_wt)
   
    return X_train_flattened



def NormalizeData(data):
          
    for n in range(data.shape[0]):
       data[n] =(data[n] - np.min(data[n])) / (np.max(data[n]) - np.min(data[n]))
    
    return data


def thresholding(data,t,original, frame):
    
    #data=predict_data_y
    #original= predict_data__arr_x
    data1= []
    name= []
    data2=data.copy()
    for n in t:
       
        data3= data2[frame].copy()
        data3[data3<n]=0
        data3[data3>=n]=1
        #data3[data3<n]=1
        print(data3.shape)
        
       # p=int(n).copy()
        data1.append(data3) 
        name.append(n)
        
        
        
        
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (1, 0, 1)
    thickness = 2
    im=[]
    
    
    image = cv2.putText(original[frame].astype("float64"), "input", org, font, 
                     fontScale, color, thickness, cv2.LINE_AA)
    image = cv2.copyMakeBorder(image, 5,5,5,5, cv2.BORDER_CONSTANT, value= [255, 255, 0])
    
    
    im.append(image)
    for q in range(len(data1)):
        image = cv2.putText(data1[q], str(name[q]), org, font, 
                         fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.copyMakeBorder(image, 5,5,5,5, cv2.BORDER_CONSTANT, value= [255, 255, 0])
        im.append(image.astype("float64"))
        
    res1 = cv2.hconcat((im[:4])) 
    res2 =cv2.hconcat((im[4:8]))
    res3 =cv2.hconcat((im[8:]))    
    res4 =cv2.vconcat((res1, res2, res3))    
    cv2.imshow("stackeyd",res4)
    cv2.waitKey(1)    
        
    # image = cv2.putText(data1, name, org, font, 
    #                     fontScale, color, thickness, cv2.LINE_AA)
    
        
    
        
        
    return data1, name
    




def thresholdingrgb(data,t,original, frame):
    
    #data=predict_data_y
    #original= predict_data__arr_x
    data1= []
    name= []
    data2=data.copy()
    for n in t:
       
        data3= data2[frame].copy()
        data3[data3<n]=0
        data3[data3>=n]=1
        #data3[data3<n]=1
        print(data3.shape)
        
       # p=int(n).copy()
        data1.append(data3) 
        name.append(n)
        
        
        
        
        
    font = cv2.FONT_HERSHEY_SIMPLEX
    org = (50, 50)
    fontScale = 1
    color = (1, 0, 1)
    thickness = 2
    im=[]
    
    cv2.imshow("original", original[frame].astype("float64"))
    cv2.waitKey(1)
    
    for q in range(len(data1)):
        image = cv2.putText(data1[q], str(name[q]), org, font, 
                         fontScale, color, thickness, cv2.LINE_AA)
        image = cv2.copyMakeBorder(image, 5,5,5,5, cv2.BORDER_CONSTANT, value= [255, 255, 0])
        im.append(image.astype("float64"))
        
    res1 = cv2.hconcat((im[:4])) 
    res2 =cv2.hconcat((im[4:8]))
    res3 =cv2.hconcat((im[8:]))    
    res4 =cv2.vconcat((res1, res2, res3))    
    cv2.imshow("stackeyd",res4)
    cv2.waitKey(1)    
        
    # image = cv2.putText(data1, name, org, font, 
    #                     fontScale, color, thickness, cv2.LINE_AA)
    
        
    
        
        
    return data1, name




def image_view():
        
    pass
    
   # For p in 
   # image = cv2.putText(image, 'OpenCV', org, font, 
                     #  fontScale, color, thickness, cv2.LINE_AA)
    #cv2.imshow(window_name, image)
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    