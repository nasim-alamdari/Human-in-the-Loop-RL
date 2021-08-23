#==========================================
# Human Feedback Collection class
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import numpy as np
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
import gym
from time import sleep
from datetime import datetime
import os
import sounddevice as sd
import os, glob
import librosa
#%gui qt5
import matplotlib.pyplot as plt
import librosa.display
import random
from random import randint
import matlab.engine
eng = matlab.engine.start_matlab()

from interface import *
from fittingEnv import *

def set_env_seed(env,seed):
    env.np_random = np.random.RandomState(seed)
    env.seed(seed)
    return env

class askHuman_realtime(object):

    def __init__(self,datetime_str, min_Datasize):
        self.minData_size = min_Datasize
        self.trijectories = []
        self.preferences = []
        self.datetime_str = datetime_str

    def add_preference(self,o0,o1,preference, compressed_audio1, compressed_audio2):
        self.preferences.append([o0,o1,preference, compressed_audio1, compressed_audio2])

    def add_trijactory(self,trijectory):
        self.trijectories.append(trijectory)
    
    def collect_preferences(self):
        if len(self.preferences) < self.minData_size:
            print("len(self.preferences) is =", len(self.preferences), " and self.minData_size = ",self.minData_size )
            return 0

        data_size = len(self.preferences) 
        print("Data size is = ", data_size)
        
        r = np.asarray(range(len(self.preferences)))
        np.random.shuffle(r)

        pref_dist = np.zeros( (data_size,)+ (2,), dtype=np.float32)  # this is when obs is not 1D
        o0_all    = np.zeros( (data_size,)+ (251,80), dtype=np.float32)  # this is when obs is not 1D
        o1_all    = np.zeros( (data_size,)+ (251,80), dtype=np.float32)  # this is when obs is not 1D
        audio_len = int(16000* 2.5)
        DRC1_all  = np.zeros((data_size,)+(audio_len,), dtype=np.float32)
        DRC2_all  = np.zeros((data_size,)+(audio_len,), dtype=np.float32)
        """o0_all    = np.zeros( (batch_size,)+ (80,80,5), dtype=np.float32)  # this is when obs is not 1D
        o1_all    = np.zeros( (batch_size,)+ (80,80,5), dtype=np.float32)  # this is when obs is not 1D"""
            
        for i in range(data_size): #r[data_size:]: # forward pass
            o0,o1,preference, audio1, audio2 = self.preferences[i]
            o0_all[i,:,:] = o0
            o1_all[i,:,:] = o1
            DRC1_all[i,:] = audio1[:audio_len]
            DRC2_all[i,:] = audio2[:audio_len]
            """o0_all[i,:,:,:] = o0
            o1_all[i,:,:,:] = o1"""
            print("for traning preference of this pair is: ", preference)
            #pref_dist = np.zeros([2],dtype=np.float32)
            if preference == 0:
                pref_dist[i,:] = [1.0,0.0] 
            elif preference == 1:
                pref_dist[i,:] = [0.0,1.0] 
            elif preference == 2:
                pref_dist[i,:] = [0.5,0.5]  
        np.savez('Dataset1_askHuman_M_realtime_p6.npz', name1=o0_all, name2=DRC1_all)
        np.savez('Dataset2_askHuman_M_realtime_p6.npz', name1=o1_all, name2=DRC2_all )
        np.save('labels_askHuman_M_realtime_p6.npy', pref_dist)
         
    def _get_two_compressed_audio (self, adjustCR1, adjustCR2):
        def npArray2Matlab(x):
            return matlab.double(x.tolist()) # casting a as list
        
        fs = 16000
        #path = './noisySpeech_files'
        path = '/Users/nasim/Documents/DRL_human_selfFitting/Audio-Generation/noisySpeech_files_new'
        filename = glob.glob(os.path.join(path, '*.wav'))
        seed = random.randint(0, (1<< 32) - 1)
        random.seed(seed)
        rnd_idx  = randint(0, np.asarray(filename).shape[0]-1)
        speech_file  = filename[rnd_idx]
        audio, fs = librosa.load(speech_file)
        if len(audio.shape) > 1:
            audio = audio[:,0]
        if fs != 16000: # resample to 16 kHz
            audio = librosa.resample(audio, fs, 16000)

        #print(audio.shape)
        audio_len = int(fs * 2.5) # only 2.5 seconds audio for reward predictor input
        audio     = audio [0:audio_len]
        newCR_1   = np.multiply(np.reshape(INITIAL_CRs, (5,)), np.reshape(adjustCR1, (5,)) ) # Updating the compratiton gains
        print("newCR1:", newCR_1)
        out_DRC1 = eng.perform_compression (npArray2Matlab(audio), npArray2Matlab(newCR_1), 16000, npArray2Matlab(INITIAL_softG), nargout=1)
        out_DRC1 = np.array(out_DRC1)
        # normalizing the compressed uadio
        audio_max1 = np.max(np.abs(out_DRC1))
        if audio_max1==0:
            audio_max1 = 1
        out_DRC1 = out_DRC1 / audio_max1
        out_DRC1 = out_DRC1.reshape(-1)
        
        newCR_2   = np.multiply(np.reshape(INITIAL_CRs, (5,)), np.reshape(adjustCR2, (5,)) ) # Updating the compratiton gains
        print("newCR2:", newCR_2)
        out_DRC2 = eng.perform_compression (npArray2Matlab(audio), npArray2Matlab(newCR_2), fs, npArray2Matlab(INITIAL_softG), nargout=1)
        out_DRC2 = np.array(out_DRC2)
        # normalizing the compressed uadio
        audio_max2 = np.max(np.abs(out_DRC2))
        if audio_max2==0:
            audio_max2 = 1
        out_DRC2 = out_DRC2 / audio_max2
        out_DRC2 = out_DRC2.reshape(-1)
        
        # Open a file to have users preferences
        f = open("human_pref_subject1_realtime.txt", "a")
        print('newCR_1 = ', newCR_1, file=f) # save the newCR_1 into .txt file
        print('newCR_2 = ', newCR_2, file=f) # save the newCR_2 into .txt file
        #print('speech_file_num = ', rnd_idx, file=f) # save the raw speech file number used into .txt file
        print("===================================")
        f.close()
        
        return out_DRC1, out_DRC2
    
    def _get_logMel (self, audio):
        
        fs = 16000
        nFilt = 80
        audio_len     = int(fs * 2.5) # only 2.5 seconds audio for reward predictor input
        DRC_seg       = audio[:audio_len]
        #MFSC_feat_DRC = librosa.feature.melspectrogram(DRC_seg, sr=self.fs, n_fft=self.frame_size, hop_length=self.overlap_size, n_mels=self.nFilt)
        MFSC_feat_DRC = librosa.feature.melspectrogram(y= DRC_seg  , sr=fs, hop_length=160, win_length=320, window='hann', n_mels=nFilt, fmax=8000)
        
        mellog_DRC = librosa.power_to_db(MFSC_feat_DRC, ref=np.max)
        # Normalization each row to [0,1]:
        for i in range(nFilt):
            arr = mellog_DRC[i,:]
            arr = arr - np.min(arr)
            safe_max = np.max(np.abs(arr))
            if safe_max==0:
                safe_max = 1
            arr = arr / safe_max
            mellog_DRC[i,:] = arr
    
        """mfsc_img_DRC  = np.zeros((nFilt, nFilt, 3))
        k = 0
        for i in range(5):
            mfsc_img_DRC[:,:,i]   = mellog_DRC[:,k:k+nFilt]
            k += nFilt"""
        
        return np.transpose(mellog_DRC)

    def ask_human(self):

        if len(self.trijectories) < 2:
            print("len(self.trijectories) = ", len(self.trijectories))
            return

        r = np.asarray(range(len(self.trijectories)))
        np.random.shuffle(r)
        #print("self.trijectories[r[-1]] = ", self.trijectories[r[-1]])
        t = [self.trijectories[r[-1]],self.trijectories[r[-2]]]

        # Prepear environments
        preference = -1 # initialize to none
        print("Please choose the preferred audio, 1 or 2. Press both for no neutral. Press Neither for no opinion.")

        while True:

            i1 =  randint(0,len(self.trijectories[r[-1]])-1)
            i2 =  randint(0,len(self.trijectories[r[-2]])-1)
            CR_set1 = self.trijectories[r[-1]][i1]
            CR_set2 = self.trijectories[r[-2]][i2]
            cnt = 0
            while  np.array_equal(CR_set1, CR_set2):
                np.random.shuffle(r)
                i1 =  randint(0,len(self.trijectories[r[-1]])-1)
                i2 =  randint(0,len(self.trijectories[r[-2]])-1)
                CR_set1 = self.trijectories[r[-1]][i1]
                CR_set2 = self.trijectories[r[-2]][i2]
            
            audio1, audio2 = self._get_two_compressed_audio (CR_set1, CR_set2)
            fs = 16000
            #########
            data, fs1 = sf.read('beep-01a.wav', dtype='float32')
            data_max = np.max(np.abs(data))
            if data_max==0:
                data_max = 1
            data = data / data_max
            data2 = 0.1*data # reduce the volume
            sd.play(data2, fs1)
            status = sd.wait()  # Wait until file is done playing
            #########
            win = AudioPlayer()
            win.show()
            win.set_audio(audio1, fs, 0)
            win.set_audio(audio2, fs, 1)
            key = win.get_response()
            win.close()

            print("preference is = ", key)
            
            if key == 0:
                preference = 0
            elif key == 1:
                preference = 1
            elif key == 2:
                preference = 2

            # Open a file to have users preferences
            f = open("human_pref_subject1_realtime.txt", "a")
            if preference == 0:
                pref_save = [1.0,0.0] 
            elif preference == 1:
                pref_save = [0.0,1.0] 
            else:
                pref_save = [0.5,0.5]   
            print('preference = ', pref_save, file=f) # save the preference into .txt file
            print('====================================', file=f)
            f.close()
            # End of saving preferences
            if preference != -1:
                break
        

        if preference != -1:
            self.add_preference(self._get_logMel(audio1),self._get_logMel(audio2),preference, audio1, audio2)

        #for i in range(len(envs)):
        #    envs[i].close()

        if preference == 0:
            print ("Audio1")
        elif preference == 1:
            print ("Audio2")
        elif preference == 2:
            print ("Neutral")
        else:
            print ("No oppinion")
            

