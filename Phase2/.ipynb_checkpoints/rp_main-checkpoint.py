#==========================================
# Main for Training Reward Predictor Model
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
import keras
from keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score, mean_squared_error
import h5py

from rp_train import *

# Load dataset    
Dataset1 = np.load('Dataset1_aug.npy')
Dataset2 = np.load('Dataset2_aug.npy')
labels = np.load('labels_aug.npy')

nFrames = Dataset1.shape[0]
nFilt = Dataset1.shape[2]
print("nFilt = ", nFilt)
print("nFrames = ", nFrames)

# ### Split dataset to train and validation set
##X_trn, X_test, y_trn, y_test = train_test_split(Dataset, labels, test_size=0.05, random_state=1)
#X_train, X_val, y_train, y_val = train_test_split(Dataset, labels, test_size=0.1, random_state=1)
nTrain = int(np.floor(nFrames*0.6))
nVal = int(np.floor(nFrames*0.2))
print("nTrain = ", nTrain)
print("nVal", nVal)
print("nTest", nFrames-nTrain-nVal)
X_train1 = Dataset1[0:nTrain-1,:,:]
X_train2 = Dataset2[0:nTrain-1,:,:]
y_train  = labels[0:nTrain-1,:]

X_val1 = Dataset1[nTrain:nVal+nTrain-1,:,:]
X_val2 = Dataset2[nTrain:nVal+nTrain-1,:,:]
y_val  = labels[nTrain:nVal+nTrain-1,:]

X_tst1 = Dataset1[nVal+nTrain:nFrames,:,:]
X_tst2 = Dataset2[nVal+nTrain:nFrames,:,:]
y_tst  = labels[nVal+nTrain:nFrames,:]

n_features = X_train1.shape[2]
n_time = X_train1.shape[1]
print(X_train1.shape)
print(y_train.shape)
print(X_val1.shape)
print(y_val.shape)
print(n_time)
print(n_features)

# ## Train the reward predictor based on preferences
trnRP = TrainRewardPredictor()
#"""
model, history, shared_model  = trnRP.train_model(X_train1, X_train2, y_train, X_val1, X_val2, y_val)

# ##show_summary_stats
trnRP.show_summary_stats(history)

# ## Save the model
model.save('./rewPred_model.h5')
shared_model.save('./rewPred_shared.h5')
del model  # deletes the existing model
del shared_model  # deletes the existing model
#"""

# Recreate the exact same model purely from the file
model = load_model('./rewPred_model.h5')
shared_model = load_model('./rewPred_shared.h5')

# ## Load data for Testing
"""test_set1 = np.load('test_set1_v3.npy')
test_set2 = np.load('test_set2_v3.npy')
test_labels  = np.load('test_labels_v3.npy')"""
test_set1 = X_tst1 
test_set2 = X_tst2
test_labels = y_tst

# ##Testing the reward predictor:
# Test pair data:
out = model.predict([test_set1, test_set2])
out_tst = np.argmax(out, axis=1)
true_tst = np.argmax(test_labels, axis=1)
print(accuracy_score(true_tst, out_tst))

#y_true = np.argmax(test_labels, axis = 1)
tst_true = test_labels[:,1]
out = shared_model.predict([test_set2])
out_tst = np.reshape(out, [-1])

y_tst = np.zeros(tst_true.shape[0])
for i in range(len(tst_true)):
    if tst_true[i] == 0.0 :
        y_tst[i] = 1
    elif tst_true[i] == 0.50:
        y_tst[i] = 2
    else:
        y_tst[i] = 3

pred_tst = np.zeros(tst_true.shape[0])
for i in range(len(tst_true)):
    if out_tst[i]< 0.33:
        pred_tst[i] = 1
    elif out_tst[i] >= 0.33 and out_tst[i]< 0.66:
        pred_tst[i] = 2
    elif out_tst[i] >= 0.66:
        pred_tst[i] = 3

print(accuracy_score(y_tst, pred_tst))
print("mse sum = ", mean_squared_error(tst_true,out_tst) )
#print("y_true = ",tst_true)
#print("y_pred = ",out_tst)

## save true_test and out_test
np.savez('test_rslt.npz', name1=tst_true, name2=out_tst)

fig = plt.figure()
plt.plot(tst_true)
plt.plot(out_tst, '-o')
plt.xlabel("Time step")
plt.ylabel("Preference")
fig.savefig('test_resutls_1.png')

fig = plt.figure()
plt.plot(np.square(np.subtract(tst_true,out_tst)),'-*')
plt.ylabel("MSE")
fig.savefig('test_resutls_rmse.png')

# # ==========================================================
