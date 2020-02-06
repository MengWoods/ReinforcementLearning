#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 13:38:52 2018

running Enduro with tensorflow

Enduro env source page: https://github.com/openai/gym/blob/master/gym/envs/atari/atari_env.py
future: add soft update
@author: Woods
"""
import tensorflow as tf
import numpy as np
import gym
import matplotlib.pyplot as plt

from preprocess import Preprocess # process state and reward
from configure import *
from DuelingDQN import *

tf.reset_default_graph() # for reseting the variable info, otherwise it cannot re run
#------------ other ---------------#
# np.random.seed(1)
# tf.set_random_seed(1)
'''########### training ###########'''

def train():    
    total_step = 0
    
    for ep in range (NUM_EPISODES):
        s = env.reset()
        s = process.process_state(s)   #(1,80,80,4)      
        ep_reward = 0
        ep_step = 0
        ep_loss = 0
        
        for st in range (NUM_EP_STEPS):
            if RENDER:
                env.render()
                
            
            a = module.choose_action(s) # Todo  # s: float32 (1, 80, 80), array
            s_, r, done, info = env.step(a) 
            s_ = process.process_state(s_)
            #r = process.process_reward(r)            
            module.remember(s, a, r, s_) 
#            if done == 1:
#                print ('done:',1)
            ep_reward += r
            ep_step += 1
            
            if (total_step > MEMEOY_SIZE) and (st % TRAIN_FREQUENCY==0):
                module.learn() # Todo        
            s = s_
            total_step += 1

            if (st == NUM_EP_STEPS - 1) or done:
                result = tf.Summary(value = [
                                #tf.Summary.Value(tag='ep_step', simple_value=ep_step),
                                tf.Summary.Value(tag='ep_reward', simple_value=ep_reward)
                                ])
                writer.add_summary(result, ep)
                print('Ep:', ep, '|ep_reward:', ep_reward,'|step:',st)
                break 
            
            # choose action with module.
            
'''########### begining ###########'''

if __name__ == '__main__':
    sess = tf.Session()
    #merged = tf.summary.merge_all() 
    env = gym.make(ENV_NAME)
    #env.seed(1)
    print('----source env info----'
          '\n action space:', env.action_space,
          '\n state shape:', env.observation_space,
          '\n reward range:', env.reward_range, # in reality, it ranges [-1, 1]
          '\n render modes:', env.metadata,
          '\n Env spec:', env.spec,
          '\n-----------------------')   
    s_dim = env.observation_space
    a_dim = env.action_space.n    
    
    process = Preprocess
#    processState = Preprocess.process_state()
#    processReward = Preprocess.process_reward()
    #print('============mark-0==============')
#    sess.run(tf.global_variables_initializer())
    module = DuelingDQN(n_actions=a_dim, n_features=[None,80,80,4], learning_rate=LR, reward_decay=GAMMA, replace_target_iter=REPLACE_TARGET, sess=sess)  
    #print('============mark0.5==============')
    writer = tf.summary.FileWriter(LOG_PATH, sess.graph)
    train()












