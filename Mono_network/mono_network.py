#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
我想用一个网络来实现 AC 的功能, 参考了 NAF 那篇文章的思想,但是目前不 work, 需要进一步探究.
Created on Tue Apr 17 01:18:35 2018
理念:单独网络输出 连续动作 a ,其他分流输出 V 和 A 然后构造 Q, 用 Q 升级网络,周而复始.
@author: menghaw1 
"""
import tensorflow as tf
import numpy as np
import os  #可适应不同系统,解决路径定位的作用
import shutil #是一种高层次的文件操作工具

import sys #为了调用上级目录的下级文件夹
sys.path.append("../..")
from env.car_env import CarEnv
tf.reset_default_graph() #为了充值 tf 生成的变量信息,否则不能重复运行

np.random.seed(1)
tf.set_random_seed(1)

MAX_EPISODES = 300  # 最大 episode
MAX_EP_STEPS = 300  # 最大步数设置
LR_A = 1e-4  # learning rate for actor
LR_C = 1e-4  # learning rate for critic
GAMMA = 0.9  # reward discount
REPLACE_ITER_A = 800
REPLACE_ITER_C = 700
MEMORY_CAPACITY = 2000  #记忆容量 原来是2000
BATCH_SIZE = 16
VAR_MIN = 0.1
RENDER = False  #开启窗口

LOAD = False  #重新训练,不载入之前训练过的

DISCRETE_ACTION = False

env = CarEnv(discrete_action=DISCRETE_ACTION)
STATE_DIM = env.state_dim #5
ACTION_DIM = env.action_dim #1
ACTION_BOUND = env.action_bound


sess = tf.Session()
writer = tf.summary.FileWriter('logsApril05/',sess.graph)


# all placeholder for tf
with tf.name_scope('S'):
    S = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s')
with tf.name_scope('R'):
    R = tf.placeholder(tf.float32, [None, 1], name='r')
with tf.name_scope('S_'):
    S_ = tf.placeholder(tf.float32, shape=[None, STATE_DIM], name='s_')


class Actor(object): # Actor 函数网络
    def __init__(self, sess, action_dim, action_bound, learning_rate, t_replace_iter , a ):
        self.sess = sess
        self.a_dim = action_dim
        self.action_bound = action_bound
        self.lr = learning_rate
        self.t_replace_iter = t_replace_iter
        self.t_replace_counter = 0
        self.a = a
        self.gamma = .995

        with tf.variable_scope('Actor'):
            # input s, output a
            self.a_,self.q = self._build_net(S, self.a, scope='eval_net', trainable=True)

            # input s_, output a, get a_ for critic
            self.a__,self.q_ = self._build_net(S_, self.a_, scope='target_net', trainable=False)

        self.e_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval_net')  #就是每次 a 的值
        self.t_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target_net')
        
        
        with tf.variable_scope('target_q'):
            self.target_q = R + self.gamma * self.q_
        with tf.variable_scope('TD_error'):
            self.loss = tf.reduce_mean(tf.squared_difference(self.target_q, self.q))
        with tf.variable_scope('C_train'):
            self.train_op = tf.train.RMSPropOptimizer(self.lr).minimize(self.loss)

    def _build_net(self, s, a ,scope, trainable): #创建网络
        with tf.variable_scope(scope):
            init_w = tf.contrib.layers.xavier_initializer()
            init_b = tf.constant_initializer(0.001)
            net = tf.layers.dense(s, 100, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l1',
                                  trainable=trainable)
            net = tf.layers.dense(net, 20, activation=tf.nn.relu,
                                  kernel_initializer=init_w, bias_initializer=init_b, name='l2',
                                  trainable=trainable)
            with tf.variable_scope('l3_a'):
                actions = tf.layers.dense(net, self.a_dim, activation=tf.nn.tanh, kernel_initializer=init_w,
                                          name='a', trainable=trainable)
                
                
                scaled_a = tf.multiply(actions, self.action_bound, name='scaled_a')  # Scale output to -action_bound to action_bound
                
                # tf.summary.histogram('actions',actions)
#                tf.summary.histogram('scaled_a',scaled_a)
            with tf.variable_scope('l3_V'):
                w3_v = tf.get_variable('w3_v', [20, 1], initializer=init_w)  #❓
                b3_v = tf.get_variable('b3_v', [1, 1], initializer=init_b)
                V = tf.matmul(net, w3_v) + b3_v
                
            with tf.variable_scope('l3_A'):
                w3_a = tf.get_variable('w3_a', [20, 20], initializer=init_w)
                b3_a = tf.get_variable('b3_a', [1, 20], initializer=init_b)
                A = tf.matmul(net, w3_a) + b3_a
                
            with tf.variable_scope('q'):
                q = V + (A - tf.reduce_mean(A, axis=1, keep_dims=True))   # Q(s,a)#求平均值.  reduce 是归约的意思 ,塌缩一个维度. , keep dims 就是保持维度.
                               
        return scaled_a,q

    def learn(self, s ,a,r,s_):   # batch update
        self.sess.run(self.train_op, feed_dict={S: s,self.a:a, R:r, S_:s_})
        if self.t_replace_counter % self.t_replace_iter == 0:
            self.sess.run([tf.assign(t, e) for t, e in zip(self.t_params, self.e_params)])
            print('Actor params changed')
        self.t_replace_counter += 1

    def choose_action(self, s, a):# 细看这一段如何选择动作,是否为连续动作
        s = s[np.newaxis, :]    # single state
        return self.sess.run(self.a_, feed_dict={S: s, self.a:a})[0]  # single action

#    def add_grad_to_graph(self, a_grads):
#        
##        tf.summary.histogram('a_grads',a_grads)#  看是不是每次都不一样 运行了,每次都不一样,
#        
#        with tf.variable_scope('policy_grads'):
#            self.policy_grads = tf.gradients(ys=self.a, xs=self.e_params, grad_ys=a_grads)# dy/dx
#
#        with tf.variable_scope('A_train'):
#            opt = tf.train.RMSPropOptimizer(-self.lr)  # (- learning rate) for ascent policy
#            self.train_op = opt.apply_gradients(zip(self.policy_grads, self.e_params)) #每次更改的是后者里面的所有参数.




class Memory(object): #存储记忆 s,a,r,s_
    def __init__(self, capacity, dims):
        self.capacity = capacity
        self.data = np.zeros((capacity, dims))
        self.pointer = 0

    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, a, [r], s_))
        index = self.pointer % self.capacity  # replace the old memory with new memory
        self.data[index, :] = transition
        self.pointer += 1

    def sample(self, n): #从记忆中采样 n 个
        assert self.pointer >= self.capacity, 'Memory has not been fulfilled'
        indices = np.random.choice(self.capacity, size=n)
        return self.data[indices, :]



# Create actor and critic.
actor = Actor(sess, ACTION_DIM, ACTION_BOUND[1], LR_A, REPLACE_ITER_A, a= [0.2])
#critic = Critic(sess, STATE_DIM, ACTION_DIM, LR_C, GAMMA, REPLACE_ITER_C, actor.a, actor.a_)
#actor.add_grad_to_graph(critic.a_grads)  # ❓  这句能保证每次都传入参数么?

M = Memory(MEMORY_CAPACITY, dims=2 * STATE_DIM + ACTION_DIM + 1)

saver = tf.train.Saver()
path = './discrete' if DISCRETE_ACTION else './continuous'

if LOAD:
    saver.restore(sess, tf.train.latest_checkpoint(path))
else:
    sess.run(tf.global_variables_initializer())
    
    
merged = tf.summary.merge_all()

'''----------------从此处开始运行----------------'''
def train():  #训练网络的主函数
    var = 2.  # control exploration  探索性设置 方差为2
    for ep in range(MAX_EPISODES):
        s = env.reset()
        ep_step = 0
        all_step = 0  # 这的 定义有问题,应该放到循环外部  ❓
        

        for t in range(MAX_EP_STEPS): #步数
        # while True:
            if RENDER:
                env.render()

            # Added exploration noise
            if t == 0 :
                a = [0.2]
            else:
                a = actor.choose_action(s,a) #根据 s 选择 a 
                   
            a = np.clip(np.random.normal(a, var), *ACTION_BOUND)    # add randomness to action selection for exploration #为动作添加探索性
            s_, r, done = env.step(a)
            M.store_transition(s, a, r, s_)

            if M.pointer > MEMORY_CAPACITY:  #记忆空间
                var = max([var*.9995, VAR_MIN])    # decay the action randomness
                b_M = M.sample(BATCH_SIZE)
                b_s = b_M[:, :STATE_DIM]
                b_a = b_M[:, STATE_DIM: STATE_DIM + ACTION_DIM]
                b_r = b_M[:, -STATE_DIM - 1: -STATE_DIM]
                b_s_ = b_M[:, -STATE_DIM:]

#                critic.learn(b_s, b_a, b_r, b_s_)  #critic 网络开始学习
                actor.learn(b_s, b_a, b_r, b_s_) #
                
    
#                if all_step % 50 == 0:
#                    result = sess.run(merged,feed_dict={S:b_s,R:b_r,S_:b_s_,critic.a:b_a})
#                    writer.add_summary(result,all_step)
                   

            s = s_  #状态更新
            ep_step += 1  #一个 episode中 step 增加一
            all_step += 1

            if done or t == MAX_EP_STEPS - 1: #每个 episode 所展示的数据
            # if done:
            
                result1=tf.Summary(value=[tf.Summary.Value(tag='ep_step',simple_value=ep_step)])
                writer.add_summary(result1,ep) 

                print('Ep:', ep,
                      '| Steps: %i' % int(ep_step),
                      '| Explore: %.2f' % var,
                      )
                break
    #保存 model
    if os.path.isdir(path): shutil.rmtree(path)
    os.mkdir(path)  #创建文件夹
    ckpt_path = os.path.join(path, 'DDPG.ckpt')
    save_path = saver.save(sess, ckpt_path, write_meta_graph=False)
    print("\nSave Model %s\n" % save_path)


def eval():  #载入曾经的网络
    env.set_fps(30)
    while True:
        s = env.reset()
        while True:
            env.render()
            a = actor.choose_action(s)
            s_, r, done = env.step(a)
            s = s_
            if done:
                break

if __name__ == '__main__':
    if LOAD:
        eval()
    else:
        train()
        
        
       
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
