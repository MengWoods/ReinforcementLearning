#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 13:02:11 2018
@author: Woods
Todo: utilize n_features argument.
"""

import numpy as np
import tensorflow as tf
from collections import deque
import random

class DuelingDQN: # Dueling network and DQN network
    
    def __init__(  # 默认参数放在前面
            self,
            n_actions,
            n_features,
            learning_rate=0.001,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=True,
            dueling=True,
            sess=None,
            ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter #隔了多少部更换 Q 值
        self.memory_size = memory_size
        self.batch_size = batch_size#随机梯度下降用到
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max   
        # total learning step
        self.learn_step_counter = 0#学习时记录多少步 ,用于判断是否更换 target net 参数
        # initialize zero memory [s, a, r, s_]
        #self.memory = np.zeros((self.memory_size, n_features * 2 + 2))#存储记忆长✖️高 建立上一行的矩阵
        self.M = deque(maxlen=self.memory_size)
        # consist of [target_net, evaluate_net]
        
        self.build_network()
        t_params = tf.get_collection('target_net_params')  #提取 target net 的参数
        e_params = tf.get_collection('eval_net_params')   #tiqu eval net 的参数
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]#把 t 的值变为 e 的值 #更新目标网络参数
        
        self.sess = sess
        self.sess.run(tf.global_variables_initializer())
        
    def build_network(self):
#        def setInitState(self,observation):
#            self.currentState = np.stack((observation, observation, observation, observation), axis = 2)        
#        def weight_variable(shape):
#            initial = tf.truncated_normal(shape, stddev = 0.01)
#            return tf.Variable(initial)        
#        def bias_variable(shape):
#            initial = tf.constant(0.01, shape = shape)
#            return tf.Variable(initial)        
        def conv2d(x, W, stride):
            return tf.nn.conv2d(x, W, strides = [1, stride, stride, 1], padding = "VALID") 
#        def max_pool_2x2(x):
#            return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = "SAME")
        
        def build_layers(s, c_name):           
            # inputs layer
            #s = tf.placeholder('float', [None, 80, 80, 4])          
#            W_conv1 = weight_variable([8,8,4,32]) #(filter_height, filter_width, in_channels, out_channels)
#            b_conv1 = bias_variable([32])           
#            W_conv2 = weight_variable([4,4,32,64])
#            b_conv2 = bias_variable([64])           
#            W_conv3 = weight_variable([3,3,64,64])
#            b_conv3 = bias_variable([64])           
#            W_fc1 = weight_variable([2304,512]) # search how to compute number of cell
#            b_fc1 = bias_variable([512])          
#            W_fc2 = weight_variable([512, self.n_actions])
#            b_fc2 = bias_variable([self.n_actions])
            
            w_init = tf.random_normal_initializer(0., 0.3)
            b_init = tf.constant_initializer(0.1)
            # input layer
            
            #stateinput = tf.placeholder('float',[None,80,80,4])
            with tf.variable_scope('conv1'):               
                w_c1 = tf.get_variable('w_c1', [8,8,4,32], initializer=w_init, collections=c_name)
                b_c1 = tf.get_variable('b_c1', [1,32], initializer=b_init, collections=c_name)
                conv1 = tf.nn.relu(conv2d(s,w_c1,4)+b_c1)
            with tf.variable_scope('conv2'):                   
                w_c2 = tf.get_variable('w_c2', [4,4,32,64], initializer=w_init, collections=c_name)
                b_c2 = tf.get_variable('b_c2', [1,64], initializer=b_init, collections=c_name)
                conv2 = tf.nn.relu(conv2d(conv1,w_c2,2)+b_c2)
            with tf.variable_scope('conv3'):                
                w_c3 = tf.get_variable('w_c3', [3,3,64,64], initializer=w_init, collections=c_name)
                b_c3 = tf.get_variable('b_c3', [1,64], initializer=b_init, collections=c_name)
                conv3 = tf.nn.relu(conv2d(conv2,w_c3,1)+b_c3)
                conv3_flat = tf.reshape(conv3,[-1,2304])
            with tf.variable_scope('fc1'):               
                w_f1 = tf.get_variable('w_f1', [2304,512], initializer=w_init, collections=c_name)
                b_f1 = tf.get_variable('b_f1', [1,512], initializer=b_init, collections=c_name)
                fc1 = tf.nn.relu(tf.matmul(conv3_flat,w_f1)+b_f1)
            #Q value
            with tf.variable_scope('fc2'):                
                w_f2 = tf.get_variable('w_f2', [512, self.n_actions], initializer=w_init, collections=c_name)
                b_f2 = tf.get_variable('b_f2', [1, self.n_actions], initializer=b_init, collections=c_name)
                out = tf.matmul(fc1, w_f2) + b_f2           
            return out      
        # -----------------build evaluate net-----------------#
        self.s = tf.placeholder(tf.float32,[None,80,80,4]) # replace it with features shape 
        self.q_target = tf.placeholder(tf.float32,[None, self.n_actions])
        
        with tf.variable_scope('eval_net'):
            c_name = ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_eval = build_layers(self.s, c_name)
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
        # -----------------build target net-----------------#
        self.s_ = tf.placeholder(tf.float32, [None,80,80,4])
        with tf.variable_scope('target_net'):
            c_name = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            self.q_next = build_layers(self.s_, c_name)           
        
    def choose_action(self, state): # state is an array , shape (1,80,80,4)
        #observation = tf.convert_to_tensor(state)
        #observation = tf.reshape(observation, [-1, 80, 80, 1]) # (batch, in_height, in_width, in_channels)            
        #observation = observation[np.newaxis, :]
        #observation = np.stack((state, state, state, state), axis = -1) # shape:(1, 80, 80, 4)
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: state})
        action = np.argmax(actions_value)
        if np.random.uniform() > self.epsilon:
            action = np.random.randint(0, self.n_actions)           
        return action
    
    def remember(self, s, a, r, s_): # s is an array (1, 80, 80, 4),
        if not hasattr(self, 'memory_counter'): # it returns true if an object has the given name attribute. vice versa.
            self.memory_counter = 0       
        # 
        s = s.reshape((80,80,4))
        s_ = s_.reshape((80,80,4))
        
        self.M.append((s, a, r, s_))
        
        #transition = np.hstack((s, [a, r], s_))# if this doesn't work, plz use deque as memory
        #index = self.memory_counter % self.memory_size
        #self.memory[index, :] = transition
        self.memory_counter += 1
        
    def learn(self): # very slow when it learning...
        # target network
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')        
        #sample_index = np.random.choice(min(self.memory_size, self.memory_counter), size=self.batch_size)
#        if self.memory_counter > self.memory_size:
#            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
#        else:
#            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        #batch_memory = self.M[sample_index] # sample_index is array([..,..,..,...])
        
        batch_memory = random.sample(self.M, self.batch_size)
        s_batch = [d[0] for d in batch_memory]
        a_batch = [d[1] for d in batch_memory]
        r_batch = [d[2] for d in batch_memory]
        s_batch_ = [d[3] for d in batch_memory]
        q_next = self.sess.run(self.q_next, feed_dict={self.s_: s_batch_})# next observation

        q_eval = self.sess.run(self.q_eval, {self.s: s_batch})
       
        q_target = q_eval.copy()
        
        batch_index = np.arange(self.batch_size, dtype=np.int32)
        
        #eval_act_index = a_batch.astype(int)
        #reward = batch_memory[:,]
        
        q_target[batch_index, a_batch] = r_batch + self.gamma * np.max(q_next, axis=1) # DQN

        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: s_batch,
                                                self.q_target: q_target})
        #print('cost:', self.cost)
        #tf.summary.scalar('cost',self.cost)
        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1
        
        #return self.cost
        
        
        
  





































      