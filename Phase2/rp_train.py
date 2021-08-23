#==========================================
# Training Reward Predictor Model
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import backend
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau, EarlyStopping
from numba import jit, cuda 
#import pickle

from rp_model import *

BATCH_SIZE = 64
EPOCH_COUNT = 500

class TrainRewardPredictor(object):
    
    def __init__(self):
        ## outputs
        self.batch_ize = BATCH_SIZE
        self.epoch_count = EPOCH_COUNT
    #@jit(target ="cuda")      
    def train_model(self, x_train1, x_train2, y_train, x_val1, x_val2, y_val):

        n_features = x_train1.shape[2]
        input_shape = (None, n_features)
        #input_shared = Input(input_shape, name='input_shared')
        #input_o1 = Input(input_shape, name='input1')
        #input_o2 = Input(input_shape, name='input2')

        RePred_model  = RewardPredictorNetwork() # initializing 
        #model, shared_model = RePred_model.conv_recurrent_model_build(input_shared, input_o1, input_o2)
        model, intermediate_layer_model = RePred_model.conv_recurrent_model_build((None, n_features))

    #     tb_callback = TensorBoard(log_dir='./logs/4', histogram_freq=1, batch_size=32, write_graph=True, write_grads=False,
    #                               write_images=False, embeddings_freq=0, embeddings_layer_names=None,
    #                               embeddings_metadata=None)
        # Save the model based on maximum validation_accuract
        checkpoint_callback = ModelCheckpoint('./weights_best.h5', monitor='val_acc', verbose=1,
                                              save_best_only=True, mode='max')
        
        # Reduce learning rate when a metric has stopped improving, waiting for 5 epochs.
        reducelr_callback = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1)
        
        # simple early stopping waiting for 150 epochs
        eStop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=150)

        callbacks_list = [eStop, checkpoint_callback, reducelr_callback]

        # Fit the model and get training history.
        print('Training...')
        #history = model.fit([x_train1,x_train2], y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT, verbose=1, callbacks=callbacks_list)
        history = model.fit([x_train1,x_train2], y_train, batch_size=BATCH_SIZE, epochs=EPOCH_COUNT,
                            validation_data=([x_val1,x_val2], y_val), verbose=1, callbacks=callbacks_list)

        ## save history 
        #with open('/trainHistoryDict', 'wb') as file_pi:
            #pickle.dump(history.history, file_pi)

        return model, history, intermediate_layer_model

    def show_summary_stats(self, history):
        # List all data in history
        print(history.history.keys())

        # Summarize history for accuracy
        fig = plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        #plt.title('model accuracy')
        plt.ylabel('Accuracy')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        fig.savefig('training_accuracy.png')

        # Summarize history for loss
        fig = plt.figure()
        plt.plot(history.history['loss'])
        plt.plot(history.history['val_loss'])
        #plt.title('model loss')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['train', 'validation'], loc='upper left')
        fig.savefig('training_loss.png')




