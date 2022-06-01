# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:18:59 2022

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
import time
from sklearn.model_selection import train_test_split

def preprocess_img(rgb_img):
        """
        Preprocessing for the image
        z-score normalize
        """
        # convert from RGB color-space to YCrCb
       # ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

        # equalize the histogram of the Y channel
       # ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

        # convert back to RGB color-space from YCrCb
        #equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
        
        if (rgb_img.ndim ==2):
            rgb_img = cv2.equalizeHist(rgb_img)
            equalized_img= rgb_img/255
        
        elif(rgb_img.ndim ==3):
            
            print(rgb_img.shape)
            
            row,col, chnl = rgb_img.shape
           
            if chnl==1:
                rgb_img = cv2.equalizeHist(rgb_img)
                equalized_img= rgb_img/255
            else:
                rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                equalized_img = cv2.equalizeHist(rgb_img)
                equalized_img= rgb_img/255
        else:
            
            print("çheck the image quality")
            

       # equalized_img=equalized_img-equalized_img.mean()
       # img[:,:,0]= (img[:,:,0] - img[:,:,0].mean()) #/ img[:,:,0].std()
        #img[:,:,1]= (img[:,:,1] - img[:,:,1].mean()) #/ img[:,:,1].std()
        #img[:,:,2]= (img[:,:,2] - img[:,:,2].mean()) #/ img[:,:,2].std()
        
        return equalized_img.astype('float16')



def preprocess_imgrgb(rgb_img):
        """
        Preprocessing for the image
        z-score normalize
        """
       
        
        if (rgb_img.ndim ==2):
            rgb_img = cv2.equalizeHist(rgb_img)
            equalized_img= rgb_img/255
        
        elif(rgb_img.ndim ==3):
            
            print(rgb_img.shape)
            
            row,col, chnl = rgb_img.shape
           
            if chnl==1:
                rgb_img = cv2.equalizeHist(rgb_img)
                equalized_img= rgb_img/255
            else:
                #rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
                
                
                ycrcb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)

               # equalize the histogram of the Y channel
                ycrcb_img[:, :, 0] = cv2.equalizeHist(ycrcb_img[:, :, 0])

                  # convert back to RGB color-space from YCrCb
                equalized_img = cv2.cvtColor(ycrcb_img, cv2.COLOR_YCrCb2BGR)
                equalized_img= equalized_img/255
        else:
            
            print("çheck the image quality")
            

       # equalized_img=equalized_img-equalized_img.mean()
       # img[:,:,0]= (img[:,:,0] - img[:,:,0].mean()) #/ img[:,:,0].std()
        #img[:,:,1]= (img[:,:,1] - img[:,:,1].mean()) #/ img[:,:,1].std()
        #img[:,:,2]= (img[:,:,2] - img[:,:,2].mean()) #/ img[:,:,2].std()
        
        return equalized_img.astype('float16')











def preprocess_label( label):
        
        label[label > 0.5] = 1.0
        label[label <= 0.5] = -1.0

        return label/255
    
    
    

def data_read(path_img, path_label, img_wt, img_ht,x_total,y_total,channel):
    i=1
    x_total.clear()
    y_total.clear()
    
    for impath in glob.glob(os.path.join(path_img,'*.png')):
        print(ntpath.basename(impath))
        img = cv2.imread(str(impath))
       # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for satellite
        resized_img = cv2.resize(img,(img_wt,img_ht), interpolation = cv2.INTER_NEAREST)
        
        if channel ==1:
            resized_img= preprocess_img(resized_img)
        else:
            resized_img= preprocess_imgrgb(resized_img)
       
        x_total.append(resized_img)
        
        y_path = os.path.join(path_label, ntpath.basename(impath))    
        img1 = cv2.imread(str(y_path))
        resized_img = cv2.resize(img1,(img_wt,img_ht), interpolation = cv2.INTER_NEAREST)
        resized_img= preprocess_img(resized_img)
        y_total.append(resized_img)
        
        if (i%500 == 0):
           # cv2.imshow("Raw",img)
           # cv2.waitKey(1)
           # cv2.imshow("mask",img1)
            res = np.hstack((img,img1)) 
            cv2.imshow("stacked",res)
            cv2.waitKey(1)
            
        i+=1
            
    if len(x_total) == len(y_total):
        print('Image samnple is matching to label data:', len(x_total),  len(y_total))
    else:
        print('Image samnple is not matching to label data, "CHECK DATA"', len(x_total),  len(y_total))
         
    cv2.destroyAllWindows()
    
    return x_total, y_total
    
    
    
def data_read_pred(path_img, img_wt, img_ht,x_total,channel):
    i=1
    for impath in glob.glob(os.path.join(path_img,'*.png')):
        print(ntpath.basename(impath))
        img = cv2.imread(str(impath))
       # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) for satellite
        resized_img = cv2.resize(img,(img_wt,img_ht), interpolation = cv2.INTER_NEAREST)
        
        
        if channel ==1:
           resized_img= preprocess_img(resized_img)
        else:
           resized_img= preprocess_imgrgb(resized_img)
        
        x_total.append(resized_img)
        
        
        
        if (i%500 == 0):
            cv2.imshow("Raw",img)
            cv2.waitKey(1)
            
        i+=1
            
        print('Check the sample size',  len(x_total))
         
    cv2.destroyAllWindows()
    
    return x_total
    
    
    
    
    


def arr_cove(x,y):
    x_sample = np.array(x, dtype='float16')
    y_sample = np.array(y,  dtype='float16')
    
    return x_sample, y_sample


def arr_cove_prd (x):
    x_sample = np.array(x, dtype='float16')
   # y_sample = np.array(y,  dtype='float16')
    
    return x_sample#, y_sample


def data_split (x_trn_dt,y_trn_dt):
    
    X_train, X_test, y_train, y_test = train_test_split(x_trn_dt,y_trn_dt,  test_size=0.20, random_state=None, shuffle=True, stratify=None)
   
    return X_train, X_test, y_train, y_test 






    
    
    
    
    
    
    
    
    
    
    
    
    
    