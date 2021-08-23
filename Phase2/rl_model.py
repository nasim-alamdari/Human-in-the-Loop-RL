#==========================================
# RL Model
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import numpy as np
import tensorflow as tf
from datetime import datetime
import os
import librosa
import matplotlib.pyplot as plt
import librosa.display

from keras.models import Sequential
from keras.layers import Dense
from keras.models import Model, Sequential
from keras.layers import Input, Dense, concatenate, Flatten, Conv2D, Permute, Reshape
from keras.layers.pooling import MaxPooling2D
from keras.callbacks import History 
from keras.utils import np_utils, multi_gpu_model

class RLModel(object):
    def __init__(self,obs_size,action_size,datetime_str,layer_sizes=[12,6,8], use_bins = False):
        self.obs_size = obs_size
        self.action_size = action_size
        self.layer_sizes = layer_sizes
        self.use_bins = use_bins
        self.lr = 0.00025
        self.lr_decay = 0.000

        self.create_model()

        self.datetime_str = datetime_str
        self.checkpoint_path = "./trainedAgent_"+datetime_str+".h5"

    def create_network(self):
        if self.use_bins:
            output_bins = 3 # three possible output
            input1 = Input(shape=(self.obs_size,))
            model = Reshape((80,80,3))(input1)
            input2 = Input(shape=(5,))
            model = Conv2D(32, (3, 3), activation='relu')(model)
            model = Conv2D(64, (3, 3), activation='relu')(model)
            model = Conv2D(128, (3, 3), activation='relu')(model)

            model = Flatten()(model)
            model = concatenate([model, input2])
            model = Dense(256, activation='relu')(model)
            model = Dense(256, activation='relu')(model)
            model_ouput = Dense(self.action_size, activation = 'softmax')(model)
            
            G1 = Dense(128)(model_ouput)
            G1_output = Dense(output_bins, activation="softmax")(G1)
            G2 = Dense(128)(model_ouput)
            G2_output = Dense(output_bins, activation="softmax")(G2)
            G3 = Dense(128)(model_ouput)
            G3_output = Dense(output_bins, activation="softmax")(G3)
            G4 = Dense(128)(model_ouput)
            G4_output = Dense(output_bins, activation="softmax")(G4)
            G5 = Dense(128)(model_ouput)
            G5_output = Dense(output_bins, activation="softmax")(G5)

            #model = Model(inputs=model_input, outputs=[G1_output, G2_output, G3_output, G4_output, G4_output, G5_output])
            model = Model(inputs=[input1, input2], outputs=[G1_output, G2_output, G3_output, G4_output, G4_output, G5_output])
        else:
            input1 = Input(shape=self.obs_size)
            input2 = Input(shape=(5,))
            #model_input = Input(shape=(self.obs_size,))
            #model = Reshape((80,80,3))(model_input)
            model = Conv2D(32, (3, 3), activation='relu', name='convolution_1')(input1)
            model = Conv2D(64, (3, 3), activation='relu', name='convolution_2')(model)
            model = Conv2D(128, (3, 3), activation='relu',name='convolution_3')(model)

            model = Flatten()(model)
            model = concatenate([model, input2])
            model = Dense(256, activation='relu', name='dense_1')(model)
            model = Dense(256, activation='relu', name='dense_2')(model)
            model_ouput = Dense(self.action_size, name='output')(model)

            #model = Model(model_input, model_ouput)
            model = Model(inputs=[input1, input2], outputs=model_ouput)       

        try:
            model = multi_gpu_model(model)
        except:
            pass
        
        print("Agent network :")
        model.summary()
        model.compile(loss='mse', optimizer='adam')

        return model
    
    def _show_summary_stats(self, history):
        # List all data in history
        print(history.history.keys())

        # Summarize history for loss
        #fig = plt.figure()
        #plt.plot(history.history['loss'])
        #plt.title('model loss')
        #plt.ylabel('loss')
        #plt.xlabel('epoch')
        #plt.legend(['train'], loc='upper left')
        #fig.savefig('training_rlAgent_loss.png')
        #plt.show()

    def loss(self,std_loss=True):
        if std_loss:
            return tf.reduce_mean(tf.square(self.target-self.prediction))
        else:
            return 1.0

    def create_model(self):
        self.graph = tf.Graph()
        with self.graph.as_default():
            #self.session = tf.Session(graph=self.graph)
            self.session = tf.compat.v1.Session(graph=self.graph)
            with self.session.as_default():
                self.model = self.create_network()

    def run(self,obs,sess):
        with self.graph.as_default():
            with self.session.as_default():
                pred = self.model.predict(obs) # gives action
        return pred

    def train(self,batch,sess):
        with self.graph.as_default():
            with self.session.as_default():
                history = self.model.fit(batch[0],batch[1],nb_epoch=1,batch_size=len(batch[0]),verbose=0)
                self.model.save("rl_model.h5")
                self._show_summary_stats(history)
                
    def save(self):		
        if not os.path.isdir(os.path.dirname(self.checkpoint_path)):
            os.mkdir(os.path.dirname(self.checkpoint_path))
        with self.graph.as_default():
            #self.model.save_weights(self.checkpoint_path)
            self.model.save("rl_model.h5")

    def load(self,datetime_str):
        with self.graph.as_default():
            self.model.load("./trainedAgent_"+datetime_str+".h5")
