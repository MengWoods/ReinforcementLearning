#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 10 16:20:24 2019

@author: menghao

My intergration version of ddpg
Runing on Gym-CartPole_v1 environment
motivated by blog Patrick Emami ddpg and morvan's ddpg
"""
import tensorflow as tf
import numpy as np
import gym
import tflearn
from collections import deque
import datetime # for taking note
import random

tf.reset_default_graph()
# ===================================
#      Hyper parameters
# ===================================
MAX_EPISODES = 10000
MAX_EP_STEPS = 200
LR_A = 0.0001
LR_C = 0.001
GAMMA = 0.99 # reward discount
TAU = 0.001  # soft replacement
MEMORY_CAPACITY = 10000000 #0000
BATCH_SIZE = 64
RENDER = False
ENV_NAME = 'CartPole-v0'
SEED = True

NOISE_LEVEL = 0. # env noise
NOISE_ENTRY = 4
noise_flag = 0

LOGPATH = 'log/ddpg'
NOTE = 'ddpg add 3 entries noise free on state'
ACTIONS = [0,1]

# ===================================
#      Functions
# ===================================
def takeNote(note): # todo: add network structure
    f = open("LOG_DDPG.txt", "a+")
    f.write("\nDDPG=====%s====\n" % datetime.datetime.now())
    f.write("env:%s, max_ep:%s, max_ep_steps:%s, LR_A:%s, LR_C:%s, GAMMA:%s, TAU:%s, Memory:%s, Batchsize:%s, NoiseLevel:%s, LogPath:%s, Seed:%s \n"
            % (ENV_NAME,MAX_EPISODES,MAX_EP_STEPS,LR_A,LR_C,GAMMA,TAU,MEMORY_CAPACITY,BATCH_SIZE,NOISE_LEVEL,LOGPATH,SEED))
    f.write(note)
    f.close()

def printenv(e):
    observation_high = e.observation_space.high
    observation_low = e.observation_space.low
    observation_shape = e.observation_space.shape
    reward_range = e.reward_range
    #action_high = e.action_space.high
    #action_low = e.action_space.low
    action_shape = e.action_space.shape
    #env_configure = e.configure
    print('\n----------env info-------------\n')
    #print(env_configure)
    print('\nobservation shape and bound:', observation_shape, observation_high, observation_low,
          '\naction shape and bound:', action_shape, #action_high, action_low,
          '\nreward range:', reward_range)
    print('-------------------------------\n')
    state_dim = observation_shape[0] #+ NOISE_ENTRY
    action_dim = 1
    #action_bound = action_high
    # ensure action bound is symmetric
    #assert (action_high == -action_low)
    action_bound = 1
    return state_dim, action_dim, action_bound

def state_noise(s,level,num_entry):
    global noise_flag
    if noise_flag <= 5000:  
        noise = np.random.uniform(low=-1,high=1,size=(num_entry,))
#        noise = np.array([np.random.uniform(-1,1),np.random.uniform(-1,1),np.random.uniform(-1,1)])# todo: change the dimensional as shape
        noise = level * noise
    if noise_flag > 5000:
        noise = s
    noise_flag += 1
    if noise_flag ==10000:
        noise_flag = 0
    
    s = np.append(s,noise,axis=0)   
    #print('noiseflag', noise_flag)
    #s = s + noise    
    return s

def build_summaries():
    episode_reward = tf.Variable(0.)
    tf.summary.scalar("Reward", episode_reward)
    episode_ave_max_q = tf.Variable(0.)
    tf.summary.scalar("Qmax_Value", episode_ave_max_q)

    summary_vars = [episode_reward, episode_ave_max_q]
    summary_ops = tf.summary.merge_all()

    return summary_ops, summary_vars
# ===================================
#      Class
# ===================================
class replayBuffer(object):
    def __init__(self, buffer_size):
        # the right side of the deque contains the most recent experience
        self.buffer_size = buffer_size
        self.count = 0
        self.buffer = deque(maxlen = buffer_size)
    
    def add(self, s, a, r, t, s2):
        experience = (s, a, r, t, s2)
        self.buffer.append(experience)
        if self.count < self.buffer_size:
            self.count += 1
    def size(self):
        return self.count
    
    def sample_batch(self, batch_size):
        batch = []
        if self.count < batch_size:
            batch = random.sample(self.buffer, self.count)
        else:
            batch = random.sample(self.buffer, batch_size)
        s_batch = np.array([_[0] for _ in batch])
        a_batch = np.array([_[1] for _ in batch])
        r_batch = np.array([_[2] for _ in batch])
        t_batch = np.array([_[3] for _ in batch])
        s2_batch = np.array([_[4] for _ in batch])
        return s_batch, a_batch, r_batch, t_batch, s2_batch  
    def clear(self):
        self.buffer.clear()
        self.count = 0
           
class DDPG(object):
    def __init__(self, sess, a_dim, s_dim, a_bound, lr_a, lr_c, tau, batch_size, gamma):
        '''-------Hyperparameters-------'''
        self.sess = sess
        self.a_dim = a_dim
        self.s_dim = s_dim
        self.a_bound = a_bound
        self.lr_a = lr_a
        self.lr_c = lr_c
        self.tau = tau
        self.batch_size = batch_size
        self.gamma = gamma
        
        '''-------actor-------'''
        # Actor network
        self.a_inputs, self.a_out, self.a_scaled_out = self.actorNetwork()
        self.a_eval_params = tf.trainable_variables()
        #self.network_params = tf.trainable_variables()
        # Target network
        self.target_inputs, self.target_out, self.a_target_scaled_out = self.actorNetwork()
        #self.target_network_params = tf.trainable_variables()[len(self.network_params):]
        # Networks parameters    
#        self.a_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/eval')
#        self.a_targ_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Actor/target') 
#        self.a_eval_params = tf.trainable_variables()
        self.a_targ_params = tf.trainable_variables()[len(self.a_eval_params):]
        
        # Op for periodically updating actor target network with online network
        self.a_update_target_network_params = \
            [self.a_targ_params[i].assign(tf.multiply(self.a_eval_params[i], self.tau) + 
                                          tf.multiply(self.a_targ_params[i], 1. - self.tau))
                for i in range(len(self.a_targ_params))]
            
        # Gradient from critic network
        self.action_gradient = tf.placeholder(tf.float32, [None, self.a_dim])  
        # Combine the gradients here
        self.unnormalized_actor_gradients = tf.gradients(self.a_scaled_out, self.a_eval_params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x: tf.div(x, self.batch_size), self.unnormalized_actor_gradients))
        # Optimization Op for actor
        self.a_optimize = tf.train.AdamOptimizer(self.lr_a).apply_gradients(zip(self.actor_gradients, self.a_eval_params))
        self.a_num_trainable_vars = len(self.a_eval_params) + len(self.a_targ_params)
        
        '''-------critic-------'''
  
        # Critic network
        self.c_inputs, self.c_action, self.c_out = self.criticNetwork()
        self.c_eval_params = tf.trainable_variables()[self.a_num_trainable_vars:]

        # Target network
        self.c_target_inputs, self.c_target_action, self.c_target_out = self.criticNetwork()            
        # Networks parameters
#        self.c_eval_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/eval')
#        self.c_targ_params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='Critic/target')
        self.c_targ_params = tf.trainable_variables()[(len(self.c_eval_params) + self.a_num_trainable_vars):]        
        # Op for periodically updating critic target network with online network
        self.c_update_target_network_params = \
            [self.c_targ_params[i].assign(tf.multiply(self.c_eval_params[i], self.tau) + 
                                          tf.multiply(self.c_targ_params[i], 1. - self.tau))
                for i in range(len(self.c_targ_params))]
        # Network target (y_i)
        self.predicted_q_value = tf.placeholder(tf.float32, [None, 1]) # one Q value for each env.step(a)
        # Optimization Op for critic
        self.c_loss = tflearn.mean_square(self.predicted_q_value, self.c_out)      
        self.c_optimize = tf.train.AdamOptimizer(self.lr_c).minimize(self.c_loss)
        # get the gradient of the net w.r.t. the action
        self.c_action_grads = tf.gradients(self.c_out, self.c_action)
        #
    
    '''-------critic functions-------'''
    def criticNetwork(self):      
        inputs = tflearn.input_data(shape=[None, self.s_dim])
        action = tflearn.input_data(shape=[None, self.a_dim])
#        tf.summary.histogram('inputs',inputs)
#        tf.summary.histogram('action',action)
        net_1 = tflearn.fully_connected(inputs, 400)
        net_1 = tflearn.layers.normalization.batch_normalization(net_1)
        net_1 = tflearn.activations.relu(net_1)
        # Add the action tensor in the 2nd hidden layer
        t1 = tflearn.fully_connected(net_1, 300)
        t2 = tflearn.fully_connected(action, 300)
        net_2 = tflearn.activation(tf.matmul(net_1, t1.W) + tf.matmul(action, t2.W) + t2.b, activation='relu')
        # Linear layer connected to 1 output representing Q(s,a)
        # Weights are init to Uniform
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003)
        out = tflearn.fully_connected(net_2, 1, weights_init=w_init)
        return inputs, action, out
    
    def c_train(self, inputs, action, predicted_q_value):
        return self.sess.run([self.c_out, self.c_optimize], feed_dict={self.c_inputs: inputs, self.c_action: action,
                                                             self.predicted_q_value: predicted_q_value})   
    def c_predict(self, inputs, action):
        return self.sess.run(self.c_out, feed_dict={self.c_inputs: inputs, self.c_action: action})
    
    def c_predict_target(self, inputs, action):
        return self.sess.run(self.c_target_out, feed_dict={self.c_target_inputs: inputs, self.c_target_action: action})
    
    def c_action_gradients(self, inputs, actions):
        return self.sess.run(self.c_action_grads, feed_dict={self.c_inputs: inputs, self.c_action: actions})
    
    def c_update_target_network(self):
        self.sess.run(self.c_update_target_network_params)
   
    '''-------actor functions-------'''
    def actorNetwork(self):       
        inputs = tflearn.input_data(shape=[None, self.s_dim]) # None for batch learning
        net_1 = tflearn.fully_connected(inputs, 400)
        net_1 = tflearn.layers.normalization.batch_normalization(net_1)
        net_1 = tflearn.activations.relu(net_1)
        net_2 = tflearn.fully_connected(net_1, 300)
        net_2 = tflearn.layers.normalization.batch_normalization(net_2)
        net_2 = tflearn.activations.relu(net_2)
        # Final layer weights are init to Uniform[-3e-3, 3e-3]
        w_init = tflearn.initializations.uniform(minval=-0.003, maxval=0.003) 
        out = tflearn.fully_connected(net_2, self.a_dim, activation='softmax', weights_init=w_init) # tanh from -1 to 1
        # Scale output to positive and negative action bound
        scaled_out = out
        print('outshape', out.shape)
        #scaled_out = tf.multiply(out, self.a_bound)
        return inputs, out, scaled_out
    
    def a_train(self, inputs, a_gradient):
        self.sess.run(self.a_optimize, feed_dict={self.a_inputs: inputs, self.action_gradient: a_gradient})
    
    def a_predict(self, inputs):
        return self.sess.run(self.a_scaled_out, feed_dict={self.a_inputs: inputs})
    
    def a_predict_target(self, inputs):
        return self.sess.run(self.a_target_scaled_out, feed_dict={self.target_inputs: inputs})
    
    def a_update_target_network(self):
        self.sess.run(self.a_update_target_network_params)
    
    def a_get_num_trainable_vars(self):
        return self.a_num_trainable_vars


# Taken from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py, which is
# based on http://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab    
class OrnsteinUhlenbeckActionNoise:
    def __init__(self, mu, sigma=0.3, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
                self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)
    
# ===================================
#      Train 
# ===================================    
def train(sess, env, module, max_ep, max_step, logpath, buffer, render, actor_noise,stateNoiseLevel,num_entry):
    summary_ops, summary_vars = build_summaries()
    sess.run(tf.global_variables_initializer())
    writer = tf.summary.FileWriter(logpath, sess.graph)
    # Initialize target network weights? why
    module.a_update_target_network()
    module.c_update_target_network()
    #replay_buffer=buffer
    # Needed to enable BatchNorm. 
    # This hurts the performance on Pendulum but could be useful
    # in other environments.
    # tflearn.is_training(True)
    #print('mark132')

    for i in range(max_ep):
        s = env.reset() 
        #s = state_noise(s, stateNoiseLevel, num_entry)
        ep_reward = 0
        ep_ave_max_q = 0
        
        for j in range(max_step):
            if render:
                env.render()
            # add exploration noise
            p_a = module.a_predict(np.reshape(s, (1, module.s_dim))) #+ actor_noise()
            a = int(np.random.choice(ACTIONS,1,[p_a, 1-p_a]))
            s2, r, terminal, info = env.step(a)
            # s2 = state_noise(s2, stateNoiseLevel, num_entry)
            buffer.add(np.reshape(s, (module.s_dim,)), a, r, terminal, np.reshape(s2, (module.s_dim,)))
            
            # Keep adding experience to the memory until there are at least minibatch size samples
            if buffer.size() > module.batch_size:
                s_batch, a_batch, r_batch, t_batch, s2_batch = buffer.sample_batch(module.batch_size)  
                
                # Calculate targets
                #a_action = module.a_predict_target(s2_batch)
                a_batch = np.reshape(a_batch,(len(a_batch),1))
                target_q = module.c_predict_target(s2_batch, module.a_predict_target(s2_batch))
                
                y_i = []
                for k in range(module.batch_size):
                    if t_batch[k]:
                        y_i.append(r_batch[k])
                    else:
                        y_i.append(r_batch[k] + module.gamma * target_q[k])
                    
                # Update the critic given the targets
                predicted_q_value, _ = module.c_train(s_batch, a_batch, np.reshape(y_i, (module.batch_size, 1)))
                ep_ave_max_q += np.amax(predicted_q_value)
                
                # Update the actor policy using the sampled gradient
                a_outs = module.a_predict(s_batch)
                grads = module.c_action_gradients(s_batch, a_outs)
                module.a_train(s_batch, grads[0])
                
                # Update target networks
                module.a_update_target_network()
                module.c_update_target_network()
            
            s = s2
            ep_reward += r
            
            if terminal or j == max_step - 1:
                
                summary_str = sess.run(summary_ops, feed_dict={summary_vars[0]: ep_reward,
                                                               summary_vars[1]: ep_ave_max_q / float(j)})
                writer.add_summary(summary_str, i)
                writer.flush()
                
                print('| Reward: {:d} | Episode: {:d} | Qmax: {:.4f} | Steps: {:d}'.format(int(ep_reward), \
                        i, (ep_ave_max_q / float(j)), j))
                break
                
# ===================================
#      Main 
# ===================================               
def main():
    with tf.Session() as sess:
        
        env = gym.make(ENV_NAME)
        #env = env.unwrapped
        s_dim, a_dim, a_bound = printenv(env)
        if SEED:       
            env.seed(1)
            tf.set_random_seed(2)
            np.random.seed(3)
            random.seed(4)
                    # sess, a_dim, s_dim, a_bound, lr_a, lr_c, tau, batch_size, gamma
        #print('mark0')
        module = DDPG(sess, a_dim, s_dim, a_bound, lr_a=LR_A, lr_c=LR_C, tau=TAU, batch_size=BATCH_SIZE, gamma=GAMMA)
        #print('mark1')
        buffer = replayBuffer(buffer_size=MEMORY_CAPACITY)
        actor_noise = OrnsteinUhlenbeckActionNoise(mu=np.zeros(a_dim))  
        #print('mark12')

        train(sess, env, module, max_ep=MAX_EPISODES, max_step=MAX_EP_STEPS, logpath=LOGPATH,
              buffer=buffer, render=RENDER, actor_noise=actor_noise, stateNoiseLevel=NOISE_LEVEL, num_entry=NOISE_ENTRY
              )
    
if __name__ == '__main__':   
    takeNote(note=NOTE)
    main()






