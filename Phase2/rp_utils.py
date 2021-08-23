#==========================================
# Utilities for Reward Predictor 
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================


import os, glob
import numpy as np
import librosa
import librosa.display
import matlab.engine
eng = matlab.engine.start_matlab()
import math
import random 
from random import randint
from numba import jit, cuda 

FS_TARGET = 16000

class RewardPredUtils(object):
    
    
    def __init__(self):
        ## outputs
        self.fs = FS_TARGET
    #@jit(target ="cuda")     
    def get_audio(self, filename):
        seed = random.randint(0,(1<<32)-1)
        random.seed(seed)
        rnd_idx = randint(0,np.asarray(filename).shape[0]-1)
        speech_file = filename[rnd_idx]
        audio, fs = librosa.load(speech_file)

        # check if it is mono or dual channel
        if len(audio.shape) > 1:
            audio = audio [:,0]
        if fs != FS_TARGET:
            audio = librosa.resample(audio, fs, FS_TARGET)
        return audio

    #@jit(target ="cuda")  
    def apply_compression (self, audio , newCR):
        INIT_softG = np.array ([7.0,11.0,11.0,12.0,18.0]) # Define 2D matrix numpy array

        def npArray2Matlab (arr):
            return matlab.double(arr.tolist()) # casting a as list

        out_DRC = eng.perform_compression (npArray2Matlab(audio), npArray2Matlab(newCR),self.fs , npArray2Matlab(INIT_softG), nargout=1 )
        out_DRC = np.array(out_DRC)

        # normalize the compressued audio (amplitude compensation)
        audio_max1 = np.max(np.abs(out_DRC))
        if audio_max1 == 0:
            audio_max1 = 1
        out_DRC = out_DRC / audio_max1
        out_DRC = out_DRC.reshape(-1)


        return out_DRC

    #@jit(target ="cuda")  
    def get_logMel (self, audio, nFilt):
        fs = 16000
        audio_len = int(fs* 2.5) # only select first 4 seconds for reward predictor
        DRC_seg = audio[:audio_len]
        MFSC_feat = librosa.feature.melspectrogram(y=DRC_seg, sr=fs, hop_length=160,
                                                  win_length=320, window='hann', n_mels=nFilt)
        mellog = librosa.power_to_db(MFSC_feat, ref=np.max)
        #mellog = np.log(MFSC_feat + 1e-9)

        ### TODO : normalize the features?!!!
        # Normalization each row to [0,1]:
        for i in range(nFilt):
            arr = mellog[i,:]
            arr = arr - np.min(arr)
            safe_max = np.max(np.abs(arr))
            if safe_max==0:
                safe_max = 1
            arr = arr / safe_max
            mellog[i,:] = arr

        """mfsc_img = np.zeros((nFilt, nFilt, 5))
        k = 0
        for i in range(5):
            mfsc_img [:,:,i] = mellog [:,k:k+nFilt]
            k += nFilt"""

        return mellog

