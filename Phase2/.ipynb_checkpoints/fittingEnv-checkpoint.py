#==========================================
# RL's Environment
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import json
import gym
from gym import spaces
import pandas as pd
import numpy as np
import itertools
import math
import matlab.engine
eng = matlab.engine.start_matlab()
import os, glob
#import matplotlib.pyplot as plt
import librosa
import librosa.display
import random
from random import randint
from gym.utils import seeding
from copy import deepcopy
import scipy

INITIAL_CRs   = np.array ([1.1, 1.2, 1.3, 1.2, 1.4]) # define initial compression ratios (CR) here
INITIAL_softG = np.array ([6.0, 11.0, 20.0, 23.0, 20.0]) # define soft gains here

class fittingEnv(gym.Env):

    def __init__(self,discreteAction = True):
    
        #% number of filters for MFSC features
        self.nFilt = 80
        #% sampling frequency
        self.fs = int(16000.0)
        #% overlap size
        self.overlap_size = int(0.01*16000)
        #% frame size
        self.frame_size   = int(0.02*16000)
        #% First step
        self.iStep = 1
        #% Compression ratio adjustment
        self.adjustVal = np.array ([1.0,1.1,1.0,1.0,1.0])
        #% Reward each time step the self-fitting DRC imprives speech quality
        self.RewardForNotFalling = 1
        #% Penalty when the self-fitting DRC fails to improve speech quality
        self.PenaltyForFalling = -1
        self.audio = None
        self.nFrames  = 3
        self.discreteAction = discreteAction
        self.count = 1
        self.seed = random.randint(0, (1<< 32) - 1)
        
        # Define action and Observation space
        self.observation_space = spaces.Box(low=-10.0, high=10.0, dtype=np.float32, shape=(self.nFilt , self.nFilt, self.nFrames))
        #self.observation_space = spaces.Box(low=-10.0, high=10.0, dtype=np.float32, shape=(self.nFilt * self.nFilt * 5,))
        
        if self.discreteAction:
            ## Define the keys and values for dictionary | keys are parameters to control | values are possible control values
            config_overrides = {
                'CR1': [1.0,4.0],
                'CR2': [1.0,4.0],
                'CR3': [1.0,4.0],
                'CR4': [1.0,4.0],
                'CR5': [1.0,4.0] }
            keys, values   = zip(*config_overrides.items())
            self.paramNum  = len(values)
            # Generate all possible values of permutation with repetition for defined above dictionary
            self.experiments  = [dict(zip(keys, v)) for v in itertools.product(*values)]
            self.nActions     = len (self.experiments) # or np.power(3,5) five bands, each has three possible values as action 
            self.action_space = spaces.Discrete(self.nActions)
        else:
            self.nActions = 5
            self.action_space      = spaces.Box(1, +4, (self.nActions,), dtype=np.float32)
        
    def _next_observation(self):
    
        ## Get a new audio segment
        self.audio = self._get_new_audio()

        ## Compute mel spectrogram features from uncompressed audio
        audio_len  = int(self.fs * 2.5) # only 2.5 seconds audio for agent input
        audio_seg  = self.audio[:audio_len]
        #MFSC_feat_audio = librosa.feature.melspectrogram(audio_seg, sr=self.fs, n_fft=self.frame_size, hop_length=self.overlap_size, n_mels=self.nFilt)
        MFSC_feat_audio = librosa.feature.melspectrogram(y= audio_seg, sr=self.fs, hop_length=self.overlap_size, 
                                                         win_length=self.frame_size, window='hann', n_mels=self.nFilt, fmax=8000)
        mellog = librosa.power_to_db(MFSC_feat_audio, ref=np.max)
        #mellog = np.log(MFSC_feat_audio + 1e-9)
        # Normalization each row to [0,1]:
        for i in range(self.nFilt):
            arr = mellog[i,:]
            arr = arr - np.min(arr)
            safe_max = np.max(np.abs(arr))
            if safe_max==0:
                safe_max = 1
            arr = arr / safe_max
            mellog[i,:] = arr
    
        ## Create Mel spectrogram images
        # nFrames  = int(np.floor(MFSC_features.shape[1]/self.nFilt))
        mfsc_img_audio = np.zeros((self.nFilt, self.nFilt, self.nFrames))
        k = 0
        for i in range(self.nFrames):
            mfsc_img_audio[:,:,i] = mellog[:,k:k+self.nFilt]
            k = k + self.nFilt
            
        # Update system states
        obs_agent = mfsc_img_audio ## TODO2 : include newCR or action as well in the next obs.
        #obs_agent = np.reshape(obs_agent,[1,-1])
        obs_RePrd = mellog #np.reshape(obs_agent,[1,-1])
        
        return obs_agent, np.transpose(obs_RePrd), self.audio
        
    def step(self, action):
        
        # Update the next state (observation) for the agent
        def npArray2Matlab(x):
            return matlab.double(x.tolist()) # casting a as list
        
        # Execute one time step within the environment
        self.take_action(action)

        obs_agent = self._next_observation()
        
        ## Perform multi-band dynamic range compression
        newCR   = np.multiply(np.reshape(INITIAL_CRs, (5,)), np.reshape(self.adjustVal, (5,)) ) # Updating the compratiton gains
        out_DRC = eng.perform_compression (npArray2Matlab(self.audio), npArray2Matlab(newCR), self.fs, npArray2Matlab(INITIAL_softG), nargout=1)
        out_DRC = np.array(out_DRC)
        # normalizing the compressed uadio
        audio_max = np.max(np.abs(out_DRC))
        if audio_max==0:
            audio_max = 1
        out_DRC = out_DRC / audio_max
        out_DRC = out_DRC.reshape(-1)
        #scipy.io.wavfile.write("temp.wav", self.fs, self.out_DRC)
        
        audio_len     = int(self.fs * 2.5) # only 2.5 seconds audio for agent input
        DRC_seg       = out_DRC [:audio_len]
        #MFSC_feat_DRC = librosa.feature.melspectrogram(DRC_seg, sr=self.fs, n_fft=self.frame_size, hop_length=self.overlap_size, n_mels=self.nFilt)
        MFSC_feat_DRC = librosa.feature.melspectrogram(y= DRC_seg  , sr=self.fs, hop_length=self.overlap_size, 
                                                       win_length=self.frame_size, window='hann', n_mels=self.nFilt, fmax=8000)
        mellog_DRC = librosa.power_to_db(MFSC_feat_DRC, ref=np.max)
        #mellog_DRC = np.log(MFSC_feat_DRC + 1e-9)
        # Normalization each row to [0,1]:
        for i in range(self.nFilt):
            arr = mellog_DRC[i,:]
            arr = arr - np.min(arr)
            safe_max = np.max(np.abs(arr))
            if safe_max==0:
                safe_max = 1
            arr = arr / safe_max
            mellog_DRC[i,:] = arr
    
        mfsc_img_DRC  = np.zeros((self.nFilt, self.nFilt, self.nFrames))
        k = 0
        for i in range(self.nFrames):
            mfsc_img_DRC[:,:,i]   = mellog_DRC[:,k:k+self.nFilt]
            k = k + self.nFilt

        #print(mfsc_img.shape) 
        obs_RePrd = mellog_DRC
        #obs_RePrd = np.reshape(obs_RePrd,[1,-1])
        
        # Compute Reward and termination condition (Done)
        # converting the environment into a single continuous episode (based on page 14 of paper)
        if self.count > 2000:
            done = True
        else:
            done = False

        if done == True:
            reward = self.PenaltyForFalling
        else:
            reward = self.RewardForNotFalling
        
        self.count+= 1

        return obs_agent[0], np.transpose(obs_RePrd), out_DRC, np.reshape(self.adjustVal, (5,)) , reward, done, {}

    def take_action(self, action):

        if self.discreteAction:
            # Extract the specific permutation correposnd to action i
            adj = np.array(list(self.experiments[action].values())) # convert dict_values to numpy 1D array
            self.adjustVal = np.reshape(adj,(self.paramNum,1)) # convert shape (x,) to (x,1), to become similar to shape of self.CR
        else: 
            for i in range(len(action)):
                ## get actual action
                if action[i] <= 1.5:
                    self.adjustVal[i] = 1
                elif (action[i] > 1.5) and (action[i] <= 2.5):
                    self.adjustVal[i] = 2
                else:
                    self.adjustVal[i] = 4
                    
        # Open a file to have users preferences (for visit 3 only)
        #f = open("RL_env.txt", "a")
        #print('newCR_from_action = ', np.reshape(self.adjustVal, (5,)), file=f) # save the newCR_1 into .txt file
        #print('====================================', file=f)
        #f.close()
             
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    
    def reset(self):
        # Reset the state of the environment to an initial state
        self.count = 1
        return self._next_observation()

    def _get_new_audio (self):
        #path = './noisySpeech_files'
        path = 'PATH_TO_NOISY_SPEECH_FILES'
        filename = glob.glob(os.path.join(path, '*.wav'))
        #seed = random.randint(0, (1<< 32) - 1)
        #random.seed(seed)
        rnd_idx  = randint(0, np.asarray(filename).shape[0]-1)
        speech_file  = filename[rnd_idx]
        audio, fs = librosa.load(speech_file)
        if len(audio.shape) > 1:
            audio = audio[:,0]
        if fs != self.fs:
            audio = librosa.resample(audio, fs, self.fs)
            
        # Open a file to have users preferences
        f = open("RL_env.txt", "a")
        print('speech_file_num = ', rnd_idx, file=f) # save the raw speech file number used into .txt file
        f.close()
        
        return audio 
    
    def render(self, mode='human', close=False):
        # Render the environment to the screen
        pass
        #print(f'CR: {self.CR}')
        #print(f'Balance: {self.balance}')
        #print(
            #f'Shares held: {self.shares_held} (Total sold: {self.total_shares_sold})')


