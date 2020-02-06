#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 18:20:06 2018

@author: Woods
"""

ENV_NAME = 'Enduro-v0'
RENDER = 1
NUM_EPISODES = 5000
NUM_EP_STEPS = 3000
MEMEOY_SIZE = 50000 # adjustable
LOG_PATH = "Enduro_log/Dueling1"

TRAIN_FREQUENCY = 2

# Hyperparameter
LR = 0.00025
GAMMA = .99
REPLACE_TARGET = 1000
BATCH_SIZE = 32

#learning_rate=0.001
#reward_decay=0.9
#e_greedy=0.9
#replace_target_iter=200
#memory_size=3000
#batch_size=32
