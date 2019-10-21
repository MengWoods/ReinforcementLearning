"""
Double DQN & Natural DQN comparison,
The Pendulum example.

View more on my tutorial page: https://morvanzhou.github.io/tutorials/
杆子立起来的时候 reward 为0,其余为负数.
Using:
Tensorflow: 1.0
gym: 0.8.0
"""


import gym
from RL_brain import DoubleDQN
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


env = gym.make('Pendulum-v0')
env = env.unwrapped#不运行这个会有错误
env.seed(1)
MEMORY_SIZE = 3000
ACTION_SPACE = 11  #把这个游戏的连续动作离散化

sess = tf.Session()
with tf.variable_scope('Natural_DQN'):#普通的 DQN
    natural_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=False, sess=sess
    )

with tf.variable_scope('Double_DQN'):#double DQN
    double_DQN = DoubleDQN(
        n_actions=ACTION_SPACE, n_features=3, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.001, double_q=True, sess=sess, output_graph=True)

sess.run(tf.global_variables_initializer()) #tf 变量初始化


def train(RL):
    total_steps = 0
    observation = env.reset()  #环境初始化
    while True:
        #if total_steps - MEMORY_SIZE > 80: env.render()  #启动环境界面

        action = RL.choose_action(observation)

        f_action = (action-(ACTION_SPACE-1)/2)/((ACTION_SPACE-1)/4)   # convert to [-2 ~ 2] float actions  把连续的动作离散化 等距离分割
        observation_, reward, done, info = env.step(np.array([f_action]))

        reward /= 10     # normalize to a range of (-1, 0). r = 0 when get upright(normalize  归一化)
        # the Q target at upright state will be 0, because Q_target = r + gamma * Qmax(s', a') = 0 + gamma * 0
        # so when Q at this state is greater than 0, the agent overestimates the Q. Please refer to the final result.

        RL.store_transition(observation, action, reward, observation_)  #和 DQN 是一样的了.

        if total_steps > MEMORY_SIZE:   # learning     这是入口   ※
            RL.learn()

        if total_steps - MEMORY_SIZE > 20000:   # stop game
            break

        observation = observation_
        total_steps += 1
    return RL.q  #返回 Q 值

q_natural = train(natural_DQN)  #从此处运行
q_double = train(double_DQN)

plt.plot(np.array(q_natural), c='r', label='natural')#所画图为 Q 值图  普通 DQN 图
plt.plot(np.array(q_double), c='b', label='double')# double DQN 图
plt.legend(loc='best') #这是设置图例,  best 是自动分配位置
plt.ylabel('Q eval')  #纵坐标
plt.xlabel('training steps')
plt.grid()
plt.show()








































