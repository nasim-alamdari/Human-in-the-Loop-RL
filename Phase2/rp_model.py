#==========================================
# Reward Predictor Model
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import keras
from keras.models import Sequential, Model
from keras.layers import Input, Dense, TimeDistributed, LSTM, Dropout, Activation, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, Flatten, Conv2D, BatchNormalization, Lambda, concatenate
from keras.layers.advanced_activations import ELU
from keras.callbacks import ModelCheckpoint, TensorBoard, ReduceLROnPlateau
from keras import backend
from keras.utils import np_utils
from keras.optimizers import Adam, RMSprop
from keras import regularizers

from tensorflow import keras
from keras import backend as K
from numba import jit, cuda 
import tensorflow as tf


L2_regularization = 0.001

class RewardPredictorNetwork(object):
    def __init__(self):
        ## outputs
        self.r1 = []
        self.r2 = []
        self.rs = []
        self.pred = []

    #@jit(target ="cuda")  
    def conv_recurrent_model_build(self, input_shape):
        print('Building model...')
        
        def get_sub_net(input_shape):
            i_input = Input(shape=input_shape, name='original_input')
            layer = Conv1D(filters=256, kernel_size=5, name='convolution_1')(i_input)
            layer = BatchNormalization(momentum=0.9)(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.5)(layer)

            layer = Conv1D(filters=128, kernel_size=3, name='convolution_2')(layer)
            layer = BatchNormalization(momentum=0.9)(layer)
            layer = Activation('relu')(layer)
            layer = Dropout(0.5)(layer)

            layer = Conv1D(filters=128, kernel_size=2, name='convolution_3')(layer)
            layer = BatchNormalization(momentum=0.9)(layer)
            layer = Activation('relu')(layer)
            layer = MaxPooling1D(4)(layer)
            layer = Dropout(0.5)(layer)

            ## LSTM Layer
            layer = Bidirectional(LSTM(128, return_sequences=True),merge_mode='concat')(layer)
            layer = Dropout(0.5)(layer)

            layer = Bidirectional(LSTM(128, return_sequences=False),merge_mode='concat')(layer)
            layer = Dropout(0.5)(layer)

            ## Dense Layer
            layer = Dense(128, name='dense1')(layer)
            layer = Dropout(0.5)(layer)
            shared_output = Dense(1, name='shared_output')(layer)
            raw_reward = Activation('sigmoid')(shared_output)
            
            return Model(i_input, [shared_output,raw_reward])

        
        shared_model = get_sub_net(input_shape)

        input_left  = Input(shape=input_shape)
        input_right = Input(shape=input_shape)

        ## Use the shared model  
        self.r1, ll = shared_model(input_left)
        self.r2, rr = shared_model(input_right)
        rs = concatenate([self.r1, self.r2])

        ## Softmax Output
        pred = Activation('softmax', name='output_realtime')(rs)
        model = Model([input_left, input_right], pred)
        
        
        # Define custom loss
        def my_loss(y_true, y_pred):
            batch_size = K.cast(tf.shape(y_pred)[0], 'float32')
            model_o0_sum = K.exp(K.sum(y_pred[:,0])/batch_size)
            model_o1_sum = K.exp(K.sum(y_pred[:,1])/batch_size)
            p_o0_o1 = model_o0_sum / (model_o0_sum + model_o1_sum)
            p_o1_o0 = model_o1_sum / (model_o1_sum + model_o0_sum)
            loss = -( (y_true[:,0]*K.log(p_o0_o1))+ (y_true[:,1]*K.log(p_o1_o0)) )
            return loss
        
        def my_categorical_accuracy(y_true, y_pred):
            return K.cast(K.equal(K.argmax(y_true, axis=-1),
                                  K.argmax(y_pred, axis=-1)),
                                  K.floatx()) 

        ## compute loss and accuracy
        #pref = K.placeholder(K.float32, shape=(None, 2))
        #my_loss = K.reduce_mean(K.categorical_crossentropy(target=pref, output=pred, from_logits=True))
                
        try:
            model = multi_gpu_model(model)
        except:
            pass
        
        opt = Adam(lr=0.001)
        model.compile(optimizer=opt,loss='categorical_crossentropy', metrics=['accuracy'])
        # model.compile(optimizer=opt,loss=my_loss, metrics=['accuracy'])
        
        intermediate_layer_model = Model(input=input_left, output=ll)
        ## outputs
        #self.r1 = r1
        #self.r2 = r2
        self.rs = rs
        self.pred = pred

        print(shared_model.summary())
        print(model.summary())
        return model, intermediate_layer_model




