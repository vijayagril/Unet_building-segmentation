# -*- coding: utf-8 -*-
"""
Created on Fri May 13 20:14:52 2022

@author: vijay
"""

import Preprocessing as pr
import DeepUnet as un
import postprocessing as pp

import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import cv2 
import os
import random
import numpy as np
import glob
import ntpath
import time





#==============================
# size of the image, need to specify here
#=================================
Img_size = 240 
channel =1
#-----------------
model_select=2 #stages: model-I- 4, model-II--4, model-III-5, Model-IV --6

"#model setup parameters"

batch_size =16
epochs = 10
n_filters_start =16
dropout_userdef =0.1
act_fun_userdef ='relu' #leaky_relu, tahn, intermediat layers

#=============================
# Processing folder drive 
#============================
path2drive = r'D:\Machine Learning\DeepUnet'

data_folder = 'Fused Dataset' # all the data 
training_img = "images1"
training_label ="labels1"
test_img = "images"
test_label ="labels"

predict_img = "predict"


img_wt = Img_size
img_ht = Img_size
img_ch = channel



path_images = os.path.join(path2drive, data_folder)
    
path_trn_img = os.path.join(path_images,training_img)
path_trn_label = os.path.join(path_images,training_label)

path_test_img = os.path.join(path_images,test_img)
path_test_label = os.path.join(path_images,test_label)

path_predict_img = os.path.join(path_images,predict_img)


trn_data_x= []
trn_data_y=[]         
# training data 
trn_data_x, trn_data_y = pr.data_read(path_trn_img,path_trn_label, img_wt, img_ht, trn_data_x,trn_data_y,channel)




# testing data 

# test_data_x=[]
# test_data_y=[]
# test_data_x, test_data_y = pr.data_read(path_test_img,path_test_label, img_wt, img_ht, test_data_x, test_data_y,channel )


# prediction data 


predict_data_x=[] 

predict_data_x = pr.data_read_pred(path_predict_img, img_wt, img_ht, predict_data_x,channel)






"converting to  numpy array "


trn_data_arr_x, trn_data_arr_y = pr.arr_cove (trn_data_x,trn_data_y)

# test_data_arr_x, test_data_arr_y = pr.arr_cove (test_data_x,test_data_y)

predict_data__arr_x = pr.arr_cove_prd (predict_data_x)



"this is for when we do not have separate testing data"


x_train, x_test, y_train, y_test= pr.data_split(trn_data_arr_x, trn_data_arr_y)

#x_train, x_test, y_train, y_test= pr.data_split(test_data_arr_x, test_data_arr_x)



def train_net(img_ht,img_wt,img_ch,x_train, y_train, epochs,batch_size, n_filters_start,dropout_userdef, act_fun_userdef, model_select):
    
    
    file_name = 'my_model'
       
    #checkpointer = tf.keras.callbacks.ModelCheckpoint('model_for_nuclei.h5', verbose=1, save_best_only=True)
    
    callbacks = [
            tf.keras.callbacks.EarlyStopping(patience=2, monitor='val_loss'),
            tf.keras.callbacks.TensorBoard(log_dir="logs\\{}".format(file_name))]
    
    if model_select ==1:    
       model = un.u_net_shallow(img_ht, img_wt, img_ch,n_filters_start,dropout_userdef,act_fun_userdef)    # model 1
    
    elif model_select ==2: 
       model = un.u_net_shallow2(img_ht, img_wt, img_ch,n_filters_start,dropout_userdef,act_fun_userdef)     # model 2
    elif model_select ==3: 
       model = un.u_net_slight_deeper(img_ht, img_wt, img_ch,n_filters_start,dropout_userdef,act_fun_userdef) # model 3

    elif model_select ==4:
       model =un.u_net_deeper(img_ht, img_wt, img_ch,n_filters_start,dropout_userdef,act_fun_userdef) # model 4
    
    else:
       print("no model selection")
    
   
    
    result = model.fit(x_train, y_train, validation_split=0.1, batch_size=batch_size, epochs=epochs, callbacks=callbacks)
    
  
    return model, result


def test(model, x_test,y_test):
    
    test_result = model.evaluate(x_test, y_test)
    return test_result

   

def predict(model, x_predict):
    
    preds_result = model.predict(x_predict)
    
   
    return preds_result





if __name__ =='__main__':


    model_unet, trn_result = train_net(img_ht,img_wt,img_ch,x_train, y_train,epochs,batch_size, n_filters_start,dropout_userdef, act_fun_userdef, model_select)
    
    test_result= test(model_unet,x_test,y_test)
    
    preds_result = predict(model_unet, predict_data__arr_x)
    
    preds_result_flttn = pp.reshaping(preds_result, img_ht, img_wt) 
    

    predict_data_y =pp.NormalizeData(preds_result_flttn)

    "specify the thrshold value to segment it for two classes"
    frame =2
    if channel==1:
    
        t= [ 0.2,0.3,0.35,0.4,0.45,  0.5,0.55,0.6,0.65,0.7,0.75]
        
        
        predict_data_y_thr, name = pp.thresholding(predict_data_y, t,predict_data__arr_x, frame )
    
    else:
        
        t= [ 0.2,0.3,0.35,0.4,0.45,  0.5,0.55,0.6,0.65,0.7,0.75,0.8]
        
        
        predict_data_y_thr, name = pp.thresholdingrgb(predict_data_y, t,predict_data__arr_x, frame )
    
    
    































