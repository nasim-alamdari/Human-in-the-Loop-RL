#==========================================
# Data Augmentation main
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from rp_augment_utils import *

# Load dataset   
print("Load all datasets and concatenate them...")
data1_1 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p1.npz')
data1_2 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p2.npz')
data1_3 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p3.npz')
data1_4 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p4.npz')
data1_5 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p5.npz')
data1_6 = np.load('./Human_Feedback_Data/Dataset1_askHuman_m_realtime_p6.npz')
data1  = np.concatenate( (data1_1['name1'], data1_2['name1'], data1_3['name1'], data1_4['name1'], data1_5['name1'],data1_6['name1']), axis=0)
#audio1 = np.concatenate( (data1_1['name2'], data1_2['name2'], data1_3['name2'], data1_4['name2'], data1_5['name2'],data1_6['name2']), axis=0)
print(data1.shape)


data2_1 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p1.npz')
data2_2 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p2.npz')
data2_3 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p3.npz')
data2_4 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p4.npz')
data2_5 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p5.npz')
data2_6 = np.load('./Human_Feedback_Data/Dataset2_askHuman_m_realtime_p6.npz')
data2  = np.concatenate( (data2_1['name1'], data2_2['name1'], data2_3['name1'], data2_4['name1'], data2_5['name1'],data2_6['name1']), axis=0)
#audio2 = np.concatenate( (data2_1['name2'], data2_2['name2'], data2_3['name2'], data2_4['name2'], data2_5['name2'],data2_6['name2']), axis=0)
print(data2.shape)


labels = np.load('labels_v1.npy')
labels_1 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p1.npy')
labels_2 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p2.npy')
labels_3 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p3.npy')
labels_4 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p4.npy')
labels_5 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p5.npy')
labels_6 = np.load('./Human_Feedback_Data/labels_askHuman_m_realtime_p6.npy')
labels = np.concatenate( (labels_1, labels_2, labels_3, labels_4, labels_5,labels_6), axis=0)


# # ==========================================================
"""
print("Balancing the data ...")
augmnt = dataAugment(data1, data2, labels, audio1, audio2)

#min_len_label = 100
min_len_label, cnt1, cnt2, cnt3 = augmnt.find_minNumLabels()
print("autual min len label = ", min_len_label)

Dataset1_v2, Dataset2_v2, DRC1_v2, DRC2_v2, labels_v2 = augmnt.equal_data (min_len_label)
#Dataset1_aug, Dataset2_aug, labels_aug = augmnt.augment_all(Dataset1_v2, Dataset2_v2, DRC1_v2, DRC2_v2, labels_v2)
        
nFrames = Dataset1_v2.shape[0]
nFilt = Dataset1_v2.shape[2]
print("nFilt = ", nFilt)
print("nFrames = ", nFrames)
"""

# # ==========================================================
print("Perform augmentation ...")
data1_aug  = np.zeros( (2*data1.shape[0], data1.shape[1],  data1.shape[2]) )
data2_aug  = np.zeros( (2*data2.shape[0], data2.shape[1],  data2.shape[2]) )
labels_aug = np.zeros( (2*labels.shape[0],labels.shape[1]) )

data1_aug[0:data1.shape[0],:,:] = data1
data2_aug[0:data2.shape[0],:,:] = data2
labels_aug[0:labels.shape[0],:] = labels

data1_aug[data1.shape[0]:2*data1.shape[0],:,:] = data2
data2_aug[data2.shape[0]:2*data2.shape[0],:,:] = data1
labels_aug[labels.shape[0]:2*labels.shape[0],0] = labels[:,1]
labels_aug[labels.shape[0]:2*labels.shape[0],1] = labels[:,0]


np.save('Dataset1_aug.npy', data1_aug)
np.save('Dataset2_aug.npy', data2_aug )
np.save('labels_aug.npy', labels_aug)


print("******* Augmentation performed on unbalanced dataset *******")

# # ==========================================================

