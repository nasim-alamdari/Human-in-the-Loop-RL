#==========================================
# Main for RL Agent
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import numpy as np
import gym
import math
import argparse

import sys
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *

import threading
from multiprocessing import  Queue
import time
from time import sleep
import sounddevice as sd
import soundfile as sf
from copy import copy

import keras
from keras.models import load_model

from rl_utils import *
from rl_model import *
from fittingEnv import *

# ### Main: Initialization
env_name = "fittingEnv"
env = fittingEnv()
obs_agent, obs_RePrd, audio = env.reset()

action_is_box = type(env.action_space) == gym.spaces.box.Box
if action_is_box:
    action_space_n = np.sum(env.action_space.shape)
else:
    action_space_n = env.action_space.n

print("obs size = ", env.observation_space.shape)
datetime_str = str(datetime.now())
rl_model   = RLModel(env.observation_space.shape,action_space_n,datetime_str,layer_sizes=[512,512,512], use_bins = False)
mini_batch = MiniBatch(env.observation_space.shape,action_space_n,batch_size=32)
"""hc_model   = HumanCritic(env.observation_space.shape,action_space_n,datetime_str, 600)"""
hc_model = load_model('./rewPred_shared.h5')

args = { 
    'load_model_datetime' : "",
    'ask_type' : 'ask_human',
    'test' : False,
}

#ask_types = ["ask_human","ask_total_reward"]
#ask_type = args['ask_type']

### Reward predictor settings
hc_ask_human_freq_episodes1 = 2 #every two episode, ask human preferences
hc_ask_human_freq_episodes2 = 100 # When agent start traning; then only every 100 episode ask human preferences
hc_train_freq = 1200  #10 segment is used for traning reward predictor

hc_loss_mean = 0
hc_loss_mean_freq = 10
hc_loss_mean_c = 0

hc_trijectory_interval = 2
hc_tricectory_c = 0
trijectory = None # create empty array
trijectory_seed = np.random.randint(2**32)
trijectory_env_name = env_name

### Agent Settings:
eps = 1.0
eps_discount_factor = 0.0005
eps_freq = 100
render_freq = 11
rl_batchSize = 50
rl_train_freq = 20 # originally was 1 # one segment is used for traning agent
rl_train_start = 33 #rl_batchSize+1 #(1* hc_train_freq)+1

save_freq = 10

#with tf.Session() as sess:
#sess.run(tf.global_variables_initializer())
sess=None
mean_reward = 0.0
mean_reward_c = 0
update_time = 1 # every one session
run_id = 0 # number of episodes (actually all is one eposodes, there is not reset condition in this project)
nIter = 0 # number of steps
sess_maxIter = 20 
max_sess_num = 300

def rewardPred_for_NasimPref (adjustCR1):
    if adjustCR1[0]==1 and adjustCR1[4]==1:
        reward = -1.0 
    elif adjustCR1[0]==1 and adjustCR1[4]==4:
        reward = 0.0
    elif adjustCR1[0]==4 and adjustCR1[4]==4:
        reward = 1.0
    elif adjustCR1[0]==4 and adjustCR1[4]==1:
        reward = 0.5

    return reward

# window allocation for human interface :
"""if QApplication.instance() is None:
    app = QApplication(sys.argv)
%gui qt5 ## enable PyQt5 event loop integration"""

while True:
    frame = 0
    done=False
    trij_total_reward = 0

    run_start = nIter

    action_strength = np.zeros([action_space_n],dtype=np.int32)
            
    ## Trijectory
    if run_id > 0 and run_id % hc_trijectory_interval == 0:
        trijectory_seed = np.random.randint(2**32)
        trijectory_env_name = env_name
        trijectory = []
        trij_obs_list = []
        #env = set_env_seed(env,trijectory_seed)
            
    ## Status update
    if run_id > 0 and run_id % update_time == 0 and nIter > rl_train_start:
        print("mean_reward_c: ", mean_reward_c)
        print("hc_loss_mean_c: ", hc_loss_mean_c)
        print ("[ Episode:",run_id," Mean-Reward:",mean_reward/float(mean_reward_c)," Epsilon:",eps,"]")
        # Open a file to save mean_rewards
        f = open("RL_reward_s2.txt", "a")
        print(mean_reward/float(mean_reward_c), file=f) 
        f.close()
        f = open("RL_eps_s2.txt", "a")
        print(eps, file=f) 
        f.close()
        mean_reward = 0.0
        mean_reward_c = 0
        
    obs_agent, obs_RePrd, audio = env.reset()
    adjustCR = np.reshape(env.adjustVal, (5,))# only for first iteration 
    while done == False:

        #x = np.reshape(obs_agent,[1,-1]) 
        #pred = rl_model.run(x,sess)
        pred_rl = rl_model.run( [np.expand_dims(obs_agent, axis=0), np.expand_dims(adjustCR, axis=0) ] ,sess)

        if np.random.uniform() < eps:
            action = np.random.randint(action_space_n)
        else:
            action = np.argmax(pred_rl)
            action_strength[action] += 1
        
        """# just for testing reward predictor, action is random (instead of using DRL agent):
        action = np.random.randint(action_space_n) """

        old_obs_agent = copy(obs_agent)
        old_obs_RePrd = copy(obs_RePrd)
        old_audio     = audio.copy()
        old_adjustCR  = copy(adjustCR)

        obs_agent, obs_RePrd, audio, adjustCR, _, done, info = env.step(action)
        
        ## Assuming reward is given for Nasim preferences
        #reward = rewardPred_for_NasimPref (adjustCR)
        reward = hc_model.predict(np.expand_dims(obs_RePrd, axis=0))[0]
        
        if nIter > rl_train_start and nIter % rl_train_freq:
            newCR   = np.multiply(np.reshape(INITIAL_CRs, (5,)), np.reshape(adjustCR, (5,)) )
            print("for CRs = ", newCR,", pred_reward is = ", reward)
        mean_reward += reward
        mean_reward_c += 1  

        if trijectory != None:
            trijectory.append([copy(old_obs_RePrd),copy(obs_RePrd), action, copy(old_audio), audio.copy(), copy(adjustCR), done]) # add audio for Pref Platform

        # Agent Training
        mini_batch.add_sample(copy(old_obs_agent),copy(obs_agent),_,copy(old_adjustCR), copy(adjustCR), action, copy(obs_RePrd), done=done ) # agent replay buffer
        
        if nIter > rl_train_start and nIter % rl_train_freq == 0:
            print("Training DRL Agent...")
            rl_model.train(mini_batch.get_batch_hc(rl_model,sess,hc_model),sess)

        eps = 0.1+(1.0-0.1)*math.exp(-eps_discount_factor*nIter)

        if done or nIter - run_start > sess_maxIter:
            break

        nIter += 1
    run_id+=1
    # Stopping criteria
    if run_id > max_sess_num:
        print("Stop training and saving RL model ...")
        #rl_model.save()
        break
        
    print("nIter = ",nIter)
    print("run_id = ", run_id)
    print("===================================")
    
    



