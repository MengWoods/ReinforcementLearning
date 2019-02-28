#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 17:41:21 2018

@author: menghaw1
"""
import numpy as np
import tensorflow as tf

class Preprocess(): # return array this step. not tensor
    def process_state(img): # input shape = (210, 160, 3), dtype: unit8
        grayscale = np.mean(img, axis=2) # shape = (210, 160), calculate mean value along the 2nd dimension
        downsample = grayscale[::2, ::2] # shape = (105, 80), sampling step = 2
        square = downsample[0:80,:]  # size (80,80)
        
        state = np.reshape(square,(1,80,80)).astype(np.float32) # shape (1,80,80)
        
        observation = np.stack((state, state, state, state), axis = -1) # shape (1,80,80,4)
        #square = tf.convert_to_tensor(square)
        #square = tf.reshape(square, [-1, 80, 80, 1])
        
        return observation
    
    def process_reward(reward):
        return np.clip(reward, -1., 1.)
    