#==========================================
# Data Augmentation class
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import os, glob
import numpy as np
import librosa
import librosa.display
import math
import random 
from random import randint
import keras
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import classification_report
#from sklearn.metrics import accuracy_score

from rp_utils import *
utils_RP = RewardPredUtils()

class dataAugment(object):
    def __init__(self,data1, data2, labels, audio1, audio2 ):
        self.Dataset1 = data1  #data1['name1']
        self.DRC1_all = audio1 #data1['name2']

        self.Dataset2 = data2  #data2['name1']
        self.DRC2_all = audio2 #data2['name2']
        
        self.labels = labels
        
        self.nFilt = self.Dataset1.shape[2]
        
    def find_minNumLabels (self):
        nFrames = len(self.labels[:,0])
        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        for i in range(len(self.labels)):
            if self.labels[i,0] == 0.0:
                cnt1 += 1
            elif self.labels[i,0] == 0.50:
                cnt2 += 1
            else:
                cnt3 += 1

        print("num of [0.0 ,1.0] labels = ", cnt1)
        print("num of [0.5 ,0.5] labels = ", cnt2)
        print("num of [1.0 ,0.0] labels = ", cnt3)
        min_len_label = min(cnt1,cnt2, cnt3)
        return min_len_label, cnt1,cnt2, cnt3


    ### Equal dataset from each labels for training and validation
    def equal_data (self,min_len_label):

        Dataset1 = self.Dataset1 
        Dataset2 = self.Dataset2 
        DRC1_all = self.DRC1_all 
        DRC2_all = self.DRC2_all 
        labels   = self.labels
        
        Dataset1_v2 = np.zeros( [3*min_len_label, Dataset1.shape[1], Dataset1.shape[2]])
        Dataset2_v2 = np.zeros( [3*min_len_label, Dataset2.shape[1], Dataset2.shape[2]])
        DRC1_v2     = np.zeros( [3*min_len_label, DRC1_all.shape[1]])
        DRC2_v2     = np.zeros( [3*min_len_label, DRC2_all.shape[1]])
        labels_v2   = np.zeros( [3*min_len_label, labels.shape[1]])

        cnt1 = 0
        cnt2 = 0
        cnt3 = 0
        idx = 0
        for i in range(len(self.labels)):
            if labels[i,0] == 0.0:
                if cnt1 < min_len_label: 
                    Dataset1_v2[idx,:,:] = Dataset1[i,:,:]
                    Dataset2_v2[idx,:,:] = Dataset2[i,:,:]
                    DRC1_v2 [idx,:] = DRC1_all[i,:]
                    DRC2_v2 [idx,:] = DRC2_all[i,:]
                    labels_v2 [idx,:] = labels[i,:]
                    idx +=1
                    cnt1 +=1

            elif labels[i,0] == 0.50:
                if cnt2 < min_len_label: 
                    Dataset1_v2[idx,:,:] = Dataset1[i,:,:]
                    Dataset2_v2[idx,:,:] = Dataset2[i,:,:]
                    DRC1_v2 [idx,:] = DRC1_all[i,:]
                    DRC2_v2 [idx,:] = DRC2_all[i,:]
                    labels_v2 [idx,:] = labels[i,:]
                    idx +=1
                    cnt2 +=1
            else:
                if cnt3 < min_len_label: 
                    Dataset1_v2[idx,:,:] = Dataset1[i,:,:]
                    Dataset2_v2[idx,:,:] = Dataset2[i,:,:]
                    DRC1_v2 [idx,:] = DRC1_all[i,:]
                    DRC2_v2 [idx,:] = DRC2_all[i,:]
                    labels_v2 [idx,:] = labels[i,:]
                    idx +=1
                    cnt3 +=1


        ### Shuffle the datasets
        nFrames = len(labels_v2[:,0])
        nums = [x for x in range(nFrames)]
        random.shuffle(nums)

        Dataset1_v2 = Dataset1_v2[nums,:,:]
        Dataset2_v2 = Dataset2_v2[nums,:,:]
        DRC1_v2     = DRC1_v2 [nums,:]
        DRC2_v2     = DRC2_v2 [nums,:]
        labels_v2   = labels_v2[nums,:]
        
        return Dataset1_v2, Dataset2_v2, DRC1_v2, DRC2_v2, labels_v2


    ## Data Augmentation using time shift, and noise injection
    def augment_all (self, Dataset1_v2, Dataset2_v2, DRC1_v2, DRC2_v2, labels_v2):
        
        """Dataset1_v2 = self.Dataset1_v2 
        Dataset2_v2 = self.Dataset2_v2 
        DRC1_v2     = self.DRC1_v2 
        DRC2_v2     = self.DRC2_v2 
        labels_v2   = self.labels_v2"""
        
        Dataset1_aug = np.zeros( [3*Dataset1_v2.shape[0], Dataset1_v2.shape[1], Dataset1_v2.shape[2]])
        Dataset2_aug = np.zeros( [3*Dataset2_v2.shape[0], Dataset2_v2.shape[1], Dataset2_v2.shape[2]])
        labels_aug   = np.zeros( [3*labels_v2.shape[0],   labels_v2.shape[1]])

        sampling_rate = 16000
        i = 0
        for j in range(Dataset1_v2.shape[0]):
            noise_factor = 0.02
            data1_manip1 = self._manipulate_noiseInject(DRC1_v2[j,:], noise_factor)
            data2_manip1 = self._manipulate_noiseInject(DRC2_v2[j,:], noise_factor)
            #print(data_manip1.shape)

            shift_direction = 'right'
            shift_max = 0.25
            data1_manip2 =  self._manipulate_ShiftT(DRC1_v2[j,:], sampling_rate, shift_max, shift_direction)
            data2_manip2 =  self._manipulate_ShiftT(DRC2_v2[j,:], sampling_rate, shift_max, shift_direction)
            #print(data_manip2.shape)

            shift_direction = 'left'
            shift_max = 0.15
            data1_manip3 =  self._manipulate_ShiftT(DRC1_v2[j,:], sampling_rate, shift_max, shift_direction)
            data2_manip3 =  self._manipulate_ShiftT(DRC2_v2[j,:], sampling_rate, shift_max, shift_direction)
            #print(data_manip2.shape)

            mellog1_1 = utils_RP.get_logMel (data1_manip1,self.nFilt)
            mellog1_2 = utils_RP.get_logMel (data1_manip2,self.nFilt)
            mellog1_3 = utils_RP.get_logMel (data1_manip3,self.nFilt)

            mellog2_1 = utils_RP.get_logMel (data2_manip1,self.nFilt)
            mellog2_2 = utils_RP.get_logMel (data2_manip2,self.nFilt)
            mellog2_3 = utils_RP.get_logMel (data2_manip3,self.nFilt)

            Dataset1_aug[i,:,:]   = Dataset1_v2[j,:,:]
            #Dataset1_aug[i+1,:,:] = np.transpose(mellog1_1)
            Dataset1_aug[i+1,:,:] = np.transpose(mellog1_2)
            Dataset1_aug[i+2,:,:] = np.transpose(mellog1_3)

            Dataset2_aug[i,:,:]   = Dataset2_v2[j,:,:]
            #Dataset2_aug[i+1,:,:] = np.transpose(mellog2_1)
            Dataset2_aug[i+1,:,:] = np.transpose(mellog2_2)
            Dataset2_aug[i+2,:,:] = np.transpose(mellog2_3)

            labels_aug[i,:]   = labels_v2[j,:]
            #labels_aug[i+1,:] = labels_v2[j,:]
            labels_aug[i+1,:] = labels_v2[j,:]
            labels_aug[i+2,:] = labels_v2[j,:]
            i += 3
            print("i = ",i)
            if i == 3*Dataset1_v2.shape[0]:
                break

        # ## Suffle the dataset
        nFrames = Dataset1_aug.shape[0]
        nums = [x for x in range(nFrames)]
        random.shuffle(nums)

        Dataset1_augment = Dataset1_aug[nums,:,:]
        Dataset2_augment = Dataset2_aug[nums,:,:]
        labels_augment   = labels_aug[nums,:]
        
        return Dataset1_augment , Dataset2_augment, labels_augment


    ### Data Augmentation for Audio to generate synthetic data to make the model better!
    #### 1. noise injection, 2. shifting time, 3. changing pitch and 4. speed
    def _manipulate_noiseInject(self, data, noise_factor):
        noise = np.random.randn(len(data))
        augmented_data = data + noise_factor * noise
        # Cast back to same data type
        augmented_data = augmented_data.astype(type(data[0]))
        return augmented_data

    def _manipulate_ShiftT(self, data, sampling_rate, shift_max, shift_direction):
        shift = np.random.randint(sampling_rate * shift_max)
        if shift_direction == 'right':
            shift = -shift
        elif shift_direction == 'both':
            direction = np.random.randint(0, 2)
            if direction == 1:
                shift = -shift
        augmented_data = np.roll(data, shift)
        # Set to silence for heading/ tailing
        if shift > 0:
            augmented_data[:shift] = 0
        else:
            augmented_data[shift:] = 0
        return augmented_data

    def _manipulate_pitch(self, data, sampling_rate, pitch_factor):
        return librosa.effects.pitch_shift(data, sampling_rate, pitch_factor)

    def _manipulate_speed(self, data, speed_factor):
        return librosa.effects.time_stretch(data, speed_factor)
    
    
    
