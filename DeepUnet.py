# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:19:24 2022

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
#from Main import img_ht , img_wt , img_ch



""" #************************** MODEL-I ****************"""

def u_net_shallow(img_ht , img_wt , img_ch,n_filters_start,dropout_userdef, act_fun_userdef):
    
 #=================================================================
    
    " Parameter for the models"
    n_filters=n_filters_start
   # droprate=0.25
    growth_factor =2
    n_classes=1
    pad ='same'
    kernal_size =(3,3)
    dropout =dropout_userdef
    act_fun =act_fun_userdef
   # max_pooling= tf.keras.layers.MaxPooling2D((2, 2))#AveragePooling2D((2, 2))
    max_pooling = tf.keras.layers.AveragePooling2D((2, 2))
    
    # Total depth of the network is 5, 4 deep layers,  
    
#===================================================================================

    


    inputs = tf.keras.layers.Input((img_ht , img_wt , img_ch))

    c1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding= pad)(inputs)
    c1 = tf.keras.layers.Dropout(dropout)(c1)
    c1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c1)
    p1 = max_pooling(c1)
    
    
    n_filters *= growth_factor
    
    c2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(p1)
    c2 = tf.keras.layers.Dropout(dropout)(c2)
    c2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c2)
    p2 = max_pooling(c2)
    
    
    n_filters *= growth_factor
    c3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(p2)
    c3 = tf.keras.layers.Dropout(dropout)(c3)
    c3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c3)
    p3 = max_pooling(c3)
    
    
    n_filters *= growth_factor
    
    c4 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(p3)
    c4 = tf.keras.layers.Dropout(dropout)(c4)
    c4 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c4)
    p4 = max_pooling(c4)
    
   
    n_filters *= growth_factor
     
    c5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(p4)
    c5 = tf.keras.layers.Dropout(dropout)(c5)
    c5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c5)
    
     
    
    n_filters //= growth_factor
    u6 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(c5)
    u6 = tf.keras.layers.concatenate([u6, c4])
    c6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(u6)
    c6 = tf.keras.layers.Dropout(dropout)(c6)
    c6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c6)
     
    
    n_filters //= growth_factor
    u7 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(c6)
    u7 = tf.keras.layers.concatenate([u7, c3])
    c7 = tf.keras.layers.Conv2D(n_filters,kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(u7)
    c7 = tf.keras.layers.Dropout(dropout)(c7)
    c7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c7)
     
    
    
    n_filters //= growth_factor
    u8 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(c7)
    u8 = tf.keras.layers.concatenate([u8, c2])
    c8 = tf.keras.layers.Conv2D(n_filters,kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(u8)
    c8 = tf.keras.layers.Dropout(dropout)(c8)
    c8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c8)
     
    
    
    n_filters //= growth_factor
    u9 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(u9)
    c9 = tf.keras.layers.Dropout(dropout)(c9)
    c9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, kernel_initializer='he_normal', padding=pad)(c9)
     
    outputs = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(c9)
    
     
    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    
    #model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    model.summary()
    
    return model
    


""" #************************** MODEL-II ****************"""

def u_net_shallow2(img_ht , img_wt , img_ch,n_filters_start,dropout_userdef,act_fun_userdef):
        
    #=========================================================================
    " Parameter for the models"
    n_filters=n_filters_start
       # droprate=0.25
    growth_factor =2
    n_classes=1
    pad ='same'
    kernal_size =(3,3)
    dropout =dropout_userdef
    act_fun =act_fun_userdef
    #pooling= tf.keras.layers.MaxPooling2D((2, 2))#AveragePooling2D((2, 2))
    pooling = tf.keras.layers.AveragePooling2D((2, 2))
    
    # Total depth of the network is 5, 4 deep layers
    
    
    droprate= tf.keras.layers.Dropout(dropout)
    
     #========================================================================
     
    inputs = tf.keras.layers.Input((img_ht, img_wt, img_ch))
   
    conv1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(inputs)
    conv1 =tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv1)
    pool1 = pooling(conv1)
    
    n_filters *= growth_factor
    
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv2)
    pool2 = pooling(conv2)
    
    n_filters *= growth_factor
    
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv3)
    pool3 = pooling(conv3)
    
    n_filters *= growth_factor
    
    
    conv4 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool3)
    conv4 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4)
    drop4 = droprate(conv4)
    pool4 = pooling(drop4)
    
    n_filters *= growth_factor


    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4)
    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv5)
    drop5 =droprate(conv5)
    
    
    n_filters //= growth_factor
#tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)
    up6 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(drop5)
    merge6 = tf.keras.layers.concatenate([drop4,up6], axis = 3)
    conv6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(merge6)
    conv6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6)
    
    
    n_filters //= growth_factor

    up7 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6)
    merge7 = tf.keras.layers.concatenate([conv3,up7], axis = 3)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(merge7)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv7)
    
    
    n_filters //= growth_factor

    up8 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv7)
    merge8 = tf.keras.layers.concatenate([conv2,up8], axis = 3)
    conv8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(merge8)
    conv8 =tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv8)
    
    
    n_filters //= growth_factor

    up9 = tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv8)
    merge9 = tf.keras.layers.concatenate([conv1,up9], axis = 3)
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(merge9)
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv9)
   
    
   
    conv9 = tf.keras.layers.Conv2D(2, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv9)
    
    
    conv10 = tf.keras.layers.Conv2D(n_classes, 1, activation = 'sigmoid')(conv9)

    model = tf.keras.Model(inputs = inputs, outputs = conv10)

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    
    model.summary()

    

    return model




""" #************************** MODEL-III ****************"""



def u_net_slight_deeper(img_ht , img_wt , img_ch,n_filters_start,dropout_userdef,act_fun_userdef):
    
    #=========================================================================
    " Parameter for the models"
    n_filters=n_filters_start
       # droprate=0.25
    growth_factor =2 # constant
    n_classes=1
    pad ='same'
    kernal_size =(3,3)
    dropout =dropout_userdef
    act_fun =act_fun_userdef
    pooling= tf.keras.layers.MaxPooling2D((2, 2))#AveragePooling2D((2, 2))
    #pooling = tf.keras.layers.AveragePooling2D((2, 2))
    
    # Total depth of the network is 6, 5 deep layers
    
    
    droprate= tf.keras.layers.Dropout(dropout)
    
     #========================================================================
    
    
     
     
     
    inputs = tf.keras.layers.Input((img_ht, img_wt, img_ch))
      #inputs = BatchNormalization()(inputs)
    conv1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv1)
    pool1 = pooling(conv1)
      #pool1 = Dropout(droprate)(pool1)
      
    n_filters *= growth_factor
     # pool1 = BatchNormalization()(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv2)
    pool2 = pooling(conv2)
    pool2 = droprate(pool2)
      
    n_filters *= growth_factor
    #  pool2 = BatchNormalization()(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv3)
    pool3 = pooling(conv3)
    pool3 = droprate(pool3)
      
    n_filters *= growth_factor
     # pool3 = BatchNormalization()(pool3)
    conv4_0 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool3)
    conv4_0 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4_0)
    pool4_1 = pooling(conv4_0)
    pool4_1 = droprate(pool4_1)
      
    n_filters *= growth_factor
      #pool4_1 = BatchNormalization()(pool4_1)
    conv4_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4_1)
    conv4_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4_1)
    pool4_2 = pooling(conv4_1)
    pool4_2 = droprate(pool4_2)
      
    n_filters *= growth_factor
    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4_2)
    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv5)
      
    n_filters //= growth_factor
     
    up6_1 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv5), conv4_1])
      
      #up6_1 = BatchNormalization()(up6_1)
    conv6_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up6_1)
    conv6_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6_1)
    conv6_1 = droprate(conv6_1)
      
    n_filters //= growth_factor
     
    up6_2 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6_1), conv4_0])
     
     # up6_2 = BatchNormalization()(up6_2)
    conv6_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up6_2)
    conv6_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6_2)
    conv6_2 = droprate(conv6_2)
      
    n_filters //= growth_factor
      
    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6_2), conv3])
      
      #up7 = BatchNormalization()(up7)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up7)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv7)
    conv7 = droprate(conv7)
      
    n_filters //= growth_factor
      
    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv7), conv2])
      
      #up8 = BatchNormalization()(up8)
    conv8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up8)
    conv8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv8)
    conv8 = droprate(conv8)
      
    n_filters //= growth_factor
    
    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv8), conv1])
     
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up9)
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv9)
      
    conv10 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
      
    model = tf.keras.Model(inputs=inputs, outputs=conv10)
       
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.BinaryAccuracy(),
                      tf.keras.metrics.FalseNegatives()])
     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.summary()
    return model









""" #************************** MODEL-IV ****************"""





def u_net_deeper(img_ht , img_wt , img_ch,n_filters_start,dropout_userdef,act_fun_userdef):
    
#=========================================================================
    " Parameter for the models"
    n_filters=n_filters_start
   # droprate=0.25
    growth_factor =2
    n_classes=1
    pad ='same'
    kernal_size =(3,3)
    dropout =dropout_userdef
    act_fun =act_fun_userdef
    pooling= tf.keras.layers.MaxPooling2D((2, 2))#AveragePooling2D((2, 2))
    #pooling = tf.keras.layers.AveragePooling2D((2, 2))
    
    # Total depth of the network is 7, 6 deep layers
    
    
    droprate= tf.keras.layers.Dropout(dropout)
    
 #========================================================================



   
    
    inputs = tf.keras.layers.Input((img_ht , img_wt , img_ch))
    conv1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(inputs)
    conv1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv1)
    pool1 = pooling(conv1)
    pool1 = droprate(pool1)
    
    n_filters *= growth_factor
   # pool1 = tf.keras.layers.BatchNormalization()(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool1)
    conv2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv2)
    pool2 =pooling(conv2)
    pool2 =droprate(pool2)
    
    n_filters *= growth_factor
   # pool2 = tf.keras.layers.BatchNormalization()(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool2)
    conv3 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv3)
    pool3 = pooling(conv3)
    pool3 = droprate(pool3)
    
    
    n_filters *= growth_factor
   # pool3 = tf.keras.layers.BatchNormalization()(pool3)
    conv4_0 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool3)
    conv4_0 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4_0)
    pool4_0 = pooling(conv4_0)
    pool4_0 = droprate(pool4_0)
    
    
    n_filters *= growth_factor
    #pool4_0 = tf.keras.layers.BatchNormalization()(pool4_0)
    conv4_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4_0)
    conv4_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4_1)
    pool4_1 = pooling(conv4_1)
    pool4_1 = droprate(pool4_1)
    
    n_filters *= growth_factor
    #pool4_1 = tf.keras.layers.BatchNormalization()(pool4_1)
    conv4_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4_1)
    conv4_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv4_2)
    pool4_2 = pooling(conv4_2)
    pool4_2 = droprate(pool4_2)
    
    
    n_filters *= growth_factor
    #pool4_2 = tf.keras.layers.BatchNormalization()(pool4_2)
    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(pool4_2)
    conv5 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv5)
    conv5 = droprate(conv5)
    
    
    n_filters //= growth_factor
    
    up6 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv5), conv4_2])
   
    #up6 = tf.keras.layers.BatchNormalization()(up6)
    conv6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up6)
    conv6 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6)
    conv6 = droprate(conv6)
    
    n_filters //= growth_factor
  
    up6_1 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6), conv4_1])
   
      
    #up6_1 = tf.keras.layers.BatchNormalization()(up6_1)
    conv6_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up6_1)
    conv6_1 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6_1)
    conv6_1 = droprate(conv6_1)
    
    n_filters //= growth_factor
    up6_2 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6_1), conv4_0])
  
    #up6_2 = tf.keras.layers.BatchNormalization()(up6_2)
    conv6_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up6_2)
    conv6_2 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv6_2)
    conv6_2 = droprate(conv6_2)
    
    n_filters //= growth_factor
    
    up7 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv6_2), conv3])
    
    #up7 = tf.keras.layers.BatchNormalization()(up7)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up7)
    conv7 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv7)
    conv7 = droprate(conv7)
    
    n_filters //= growth_factor
    up8 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv7), conv2])
    
    #up8 = tf.keras.layers.BatchNormalization()(up8)
    conv8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up8)
    conv8 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv8)
    conv8 = droprate(conv8)
    
    n_filters //= growth_factor
   
    up9 = tf.keras.layers.concatenate([tf.keras.layers.Conv2DTranspose(n_filters, (2, 2), strides=(2, 2), padding=pad)(conv8), conv1])
   
      
    #up9 = tf.keras.layers.BatchNormalization()(up9)
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(up9)
    conv9 = tf.keras.layers.Conv2D(n_filters, kernal_size, activation=act_fun, padding=pad, kernel_initializer = 'he_normal')(conv9)
    
    
    conv10 = tf.keras.layers.Conv2D(n_classes, (1, 1), activation='sigmoid')(conv9)
    
    model = tf.keras.Model(inputs=inputs, outputs=conv10)   
  
    
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss=tf.keras.losses.BinaryCrossentropy(),
              metrics=[tf.keras.metrics.BinaryAccuracy(),
                        tf.keras.metrics.FalseNegatives()])
    model.summary() 
   
    
    return model























 