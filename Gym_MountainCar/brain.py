#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 21:42:39 2018
All 算法的核心部分整合在这个文件中
@author: menghaw1
"""
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

np.random.seed(1)
tf.set_random_seed(1)

# deep Q network off policy
class DeepQNetwork:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.01,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=300,
            memory_size=500,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            ):
        self.n_actions=n_actions
        self.n_features=n_features
        self.lr=learning_rate
        self.gamma=reward_decay
        self.epsilon_max=e_greedy
        self.replace_target_iter=replace_target_iter
        self.memory_size=memory_size
        self.batch_size=batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon=0 if e_greedy_increment is not None else self.epsilon_max
        
        self.learn_step_counter=0 # total learning step
        # initialize zero memory [s, a, r, s_]   此时状态,下一刻状态 加上 r 和 a
        self.memory = np.zeros((self.memory_size, n_features*2+2)) 
        
        self._build_net()
        t_params = tf.get_collection('target_net_params') #Returns a list of values in the collection with the given name
        e_params = tf.get_collection('eval_net_params')
        
        self.replace_target_op = [tf.assign(t,e) for t, e in zip(t_params,e_params)] #不清楚什么意思❓更换目标函数权重参数
        self.sess = tf.Session()
        
        if output_graph:
            tf.summary.FileWriter("logs/",self.sess.graph)
        
        self.sess.run(tf.global_variables_initializer())
        self.cost_his=[]
        #self.reward_his=[]#wu
    def _build_net(self):
        # --------------------------------build evaluate_net--------------------------------------
        self.s = tf.placeholder(tf.float32,[None,self.n_features],name='s') #input
        self.q_target = tf.placeholder(tf.float32,[None,self.n_actions],name='Q_target') #for calculating loss
        with tf.variable_scope('eval_net'):##第一层的神经元数量10
            c_names,n_l1,w_initializer,b_initializer = \
                ['eval_net_params',tf.GraphKeys.GLOBAL_VARIABLES],\
                10,\
                tf.random_normal_initializer(0.,0.3),\
                tf.constant_initializer(0.1)     #config of layers 
        
            #1st layer.
            with tf.variable_scope('l1'):
                w1=tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s,w1)+b1)
                
            #2nd layer.
            with tf.variable_scope('l2'):
                w2=tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_eval=tf.matmul(l1,w2)+b2  #只有一层中间网络,第二层直接是输出层
            
        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target,self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)
            
        #--------------------------------build target_net--------------------------------
        self.s_ = tf.placeholder(tf.float32,[None,self.n_features],name='s_')
        
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]
            
            with tf.variable_scope('l1'):
                w1=tf.get_variable('w1',[self.n_features,n_l1],initializer=w_initializer,collections=c_names)
                b1=tf.get_variable('b1',[1,n_l1],initializer=b_initializer,collections=c_names)
                l1=tf.nn.relu(tf.matmul(self.s_,w1)+b1)
            with tf.variable_scope('l2'):
                w2=tf.get_variable('w2',[n_l1,self.n_actions],initializer=w_initializer,collections=c_names)
                b2=tf.get_variable('b2',[1,self.n_actions],initializer=b_initializer,collections=c_names)
                self.q_next=tf.matmul(l1,w2)+b2 
    
    def store_transition(self,s,a,r,s_):
        if not hasattr(self,'memory_counter'):  #判断前者对象的后者属性是否存在,若存在返回 True
            self.memory_counter = 0
            
        transition = np.hstack((s,[a,r],s_)) #horizontal stack 水平合并数组.对应的是 vstack
        #replace the old memory with new memory
        index = self.memory_counter % self.memory_size
        self.memory[index,:] = transition
        self.memory_counter += 1
        
    def choose_action(self,observation):
        observation = observation[np.newaxis,:]
        if np.random.uniform()<self.epsilon:
             actions_value=self.sess.run(self.q_eval,feed_dict={self.s:observation})
             action = np.argmax(actions_value)
        else:
             action = np.random.randint(0,self.n_actions)
        return action
     
    def learn(self):
        #check to repalce target parameters. 每隔多久更换一次目标权重
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            #ghcs = self.learn_step_counter // self.replace_target_iter #向下取整, wu 加  更换次数; 也可以用 round 函数四舍五入
            #print(ghcs,' times target_params_replaced \n')
        # sample batch memory from all memory
        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size,size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter,size=self.batch_size)
        batch_memory = self.memory[sample_index,:]
    
        q_next,q_eval = self.sess.run([self.q_next,self.q_eval], #❓ 不懂这个一步骤的函数
                                                  feed_dict={ 
                    self.s_:batch_memory[:,-self.n_features:], #fixed params
                    self.s :batch_memory[:,:self.n_features], #newest params
                })
        # change q_target w.r.t q_eval's action
        q_target = q_eval.copy()
        
        batch_index=np.arange(self.batch_size,dtype=np.int32)
        eval_act_index=batch_memory[:,self.n_features].astype(int)
        reward = batch_memory[:,self.n_features+1]
        
        q_target[batch_index,eval_act_index]= reward + self.gamma * np.max(q_next,axis=1)   #q_target
        
        # train eval network
        _,self.cost = self.sess.run([self._train_op,self.loss],
                                    feed_dict={self.s:batch_memory[:,:self.n_features],
                                               self.q_target: q_target})
        
        self.cost_his.append(self.cost)
        #self.reward_his.append(reward)
        
        #increasing epsilon
        self.epsilon=self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1 
    
    def plot_cost(self):        
        fig_cost=plt.figure('fig_cost')
        plt.plot(np.arange(len(self.cost_his)),self.cost_his) 
        plt.ylabel('cost')
        plt.xlabel('training steps')
        plt.show(fig_cost)    
''' 有问题        
    def plot_reward(self):        
        plt.plot(np.arange(len(self.reward_his)),self.reward_his)
        plt.ylabel('reward')
        plt.xlabel('training steps')
        plt.show
'''    

'''---------------------------------------------------------------------'''
class DoubleDQN:
    def __init__(
            self,
            n_actions,
            n_features,
            learning_rate=0.005,
            reward_decay=0.9,
            e_greedy=0.9,
            replace_target_iter=200,
            memory_size=3000,
            batch_size=32,
            e_greedy_increment=None,
            output_graph=False,
            double_q=True,# double DQN
            sess=None,
    ):
        self.n_actions = n_actions
        self.n_features = n_features
        self.lr = learning_rate
        self.gamma = reward_decay
        self.epsilon_max = e_greedy
        self.replace_target_iter = replace_target_iter
        self.memory_size = memory_size
        self.batch_size = batch_size
        self.epsilon_increment = e_greedy_increment
        self.epsilon = 0 if e_greedy_increment is not None else self.epsilon_max

        self.double_q = double_q    # decide to use double q or not

        self.learn_step_counter = 0
        self.memory = np.zeros((self.memory_size, n_features*2+2))
        self._build_net()
        t_params = tf.get_collection('target_net_params')
        e_params = tf.get_collection('eval_net_params')
        self.replace_target_op = [tf.assign(t, e) for t, e in zip(t_params, e_params)]

        if sess is None:
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        else:
            self.sess = sess
        if output_graph:
            tf.summary.FileWriter("logs/", self.sess.graph)  #输出 board
        self.cost_his = []

    def _build_net(self):
        def build_layers(s, c_names, n_l1, w_initializer, b_initializer):
            with tf.variable_scope('l1'):
                w1 = tf.get_variable('w1', [self.n_features, n_l1], initializer=w_initializer, collections=c_names)
                b1 = tf.get_variable('b1', [1, n_l1], initializer=b_initializer, collections=c_names)
                l1 = tf.nn.relu(tf.matmul(s, w1) + b1)

            with tf.variable_scope('l2'):
                w2 = tf.get_variable('w2', [n_l1, self.n_actions], initializer=w_initializer, collections=c_names)
                b2 = tf.get_variable('b2', [1, self.n_actions], initializer=b_initializer, collections=c_names)
                out = tf.matmul(l1, w2) + b2
            return out
        # ------------------ build evaluate_net ------------------
        self.s = tf.placeholder(tf.float32, [None, self.n_features], name='s')  # input
        self.q_target = tf.placeholder(tf.float32, [None, self.n_actions], name='Q_target')  # for calculating loss

        with tf.variable_scope('eval_net'):
            c_names, n_l1, w_initializer, b_initializer = \
                ['eval_net_params', tf.GraphKeys.GLOBAL_VARIABLES], 20, \
                tf.random_normal_initializer(0., 0.3), tf.constant_initializer(0.1)  # config of layers

            self.q_eval = build_layers(self.s, c_names, n_l1, w_initializer, b_initializer)

        with tf.variable_scope('loss'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.q_target, self.q_eval))
        with tf.variable_scope('train'):
            self._train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

        # ------------------ build target_net ------------------
        self.s_ = tf.placeholder(tf.float32, [None, self.n_features], name='s_')    # input
        with tf.variable_scope('target_net'):
            c_names = ['target_net_params', tf.GraphKeys.GLOBAL_VARIABLES]

            self.q_next = build_layers(self.s_, c_names, n_l1, w_initializer, b_initializer)

    def store_transition(self, s, a, r, s_):
        if not hasattr(self, 'memory_counter'):
            self.memory_counter = 0
        transition = np.hstack((s, [a, r], s_))
        index = self.memory_counter % self.memory_size
        self.memory[index, :] = transition
        self.memory_counter += 1

    def choose_action(self, observation):  #从此处入口
        observation = observation[np.newaxis, :]
        actions_value = self.sess.run(self.q_eval, feed_dict={self.s: observation})
        action = np.argmax(actions_value)

        if not hasattr(self, 'q'):  # record action value it gets  记录选择的 Qmax 值
            self.q = []
            self.running_q = 0
        self.running_q = self.running_q*0.99 + 0.01 * np.max(actions_value)
        self.q.append(self.running_q)

        if np.random.uniform() > self.epsilon:  # choosing action随机
            action = np.random.randint(0, self.n_actions)
        return action

    def learn(self):
        #这一段和 DQN 一样
        if self.learn_step_counter % self.replace_target_iter == 0:
            self.sess.run(self.replace_target_op)
            print('\ntarget_params_replaced\n')

        if self.memory_counter > self.memory_size:
            sample_index = np.random.choice(self.memory_size, size=self.batch_size)
        else:
            sample_index = np.random.choice(self.memory_counter, size=self.batch_size)
        batch_memory = self.memory[sample_index, :]

        #这一段和 DQN 不一样
        q_next, q_eval4next = self.sess.run(   #一个是 Qnext 神经网络,一个是 Qevlauation 神经网络 后者是现实中用 Q 估计出来的值
            [self.q_next, self.q_eval],
            feed_dict={self.s_: batch_memory[:, -self.n_features:],    # next observation
                       self.s: batch_memory[:, -self.n_features:]})    # next observation
        q_eval = self.sess.run(self.q_eval, {self.s: batch_memory[:, :self.n_features]})  #t 时刻真正的值

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_size, dtype=np.int32)
        eval_act_index = batch_memory[:, self.n_features].astype(int)
        reward = batch_memory[:, self.n_features + 1]

        if self.double_q:# 最大的不同和 DQN 相比   如果是 double 时候
            max_act4next = np.argmax(q_eval4next, axis=1)        # the action that brings the highest value is evaluated by q_eval
            # 下, DDQN选择 q_next 依据 q_eval 选出的动作. # 上,q_eval 得出的最高奖励动作.
            selected_q_next = q_next[batch_index, max_act4next]  # Double DQN, select q_next depending on above actions
        else:  #如果是普通 DQN
            selected_q_next = np.max(q_next, axis=1)    # the natural DQN

        q_target[batch_index, eval_act_index] = reward + self.gamma * selected_q_next
        #下面和 DQN 一样
        _, self.cost = self.sess.run([self._train_op, self.loss],
                                     feed_dict={self.s: batch_memory[:, :self.n_features],
                                                self.q_target: q_target})
        self.cost_his.append(self.cost)

        self.epsilon = self.epsilon + self.epsilon_increment if self.epsilon < self.epsilon_max else self.epsilon_max
        self.learn_step_counter += 1







































        
        











































































