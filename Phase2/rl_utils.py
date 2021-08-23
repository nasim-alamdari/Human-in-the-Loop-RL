#==========================================
# Utilities for RL Agent 
# Author: Nasim Alamdari
# Date:   Dec. 2020
#==========================================

import numpy as np
import tensorflow as tf

class MiniBatch(object):
    def __init__(self,obs_size,actions,batch_size=10):
        self.sample_obs = []
        self.sample_ys = []
        self.gamma = 0.99
        self.obs_size = obs_size
        self.actions = actions
        self.batch_size = batch_size
        self.capacity = 100000
        self.samples = []

    def add_sample(self,obs, future_obs, reward, old_adjustCR, adjustCR, action, future_obs_RePrd, done=False):  # replay buffer
        self.samples.append([obs, future_obs, reward, old_adjustCR, adjustCR, action, future_obs_RePrd, done])
        if len(self.samples) > self.capacity:
            del self.samples[0]
    
    def _rewardPred_for_NasimPref (self,adjustCR1):
        if adjustCR1[0]==1 and adjustCR1[4]==1:
            reward = -1.0 
        elif adjustCR1[0]==1 and adjustCR1[4]==4:
            reward = 0.0
        elif adjustCR1[0]==4 and adjustCR1[4]==4:
            reward = 1
        elif adjustCR1[0]==4 and adjustCR1[4]==1:
            reward = 0.5

        return reward


    def get_batch_hc(self,model,sess,hc_model):
        r = np.arange(0,len(self.samples))
        np.random.shuffle(r)
        batch_size = int(min(self.batch_size,len(self.samples)))
        batch = np.asarray(self.samples)[r[:batch_size]]

        states = np.array([sample[0] for sample in batch],dtype=np.float32) # from [obs,future_obs, reward, action ,done], get obs
        future_states = np.array([(np.zeros(self.obs_size) if sample[7] else sample[1]) for sample in batch],dtype=np.float32) # from [obs,future_obs, reward, action ,done], if done =False: get future_obs

        adjustCR        = np.array([sample[3] for sample in batch],dtype=np.float32)
        future_adjustCR = np.array([(np.zeros(5) if sample[7] else sample[4]) for sample in batch],dtype=np.float32)

        q_value_batch = model.run([states, adjustCR ],sess) # predict the rl agent (get action) -> one-hot encoding labels
        future_q_value_batch = model.run([future_states, future_adjustCR ],sess) # predict the rl agent (get action) -> one-hot encoding labels

        #x = np.zeros([batch_size,self.obs_size],dtype=np.float32)
        x1 = np.zeros( (batch_size,)+ self.obs_size, dtype=np.float32)  # this is when obs is not 1D
        x2 = np.zeros( [batch_size,5], dtype=np.float32)  # 5 action for 5 gains
        y  = np.zeros([batch_size,self.actions],dtype=np.float32)       
        #y = np.zeros( (batch_size,)+ self.actions , dtype=np.float32) # this is when action is not 1D

        for i in range(batch_size):
            state,future_state,_,old_adjustCR, adjustCR, action, state_rePrd, done = batch[i]
            """#reward = hc_model.predict(np.reshape(state_rePrd,[1,-1]))[0] """
            reward = hc_model.predict(np.expand_dims(state_rePrd, axis=0))[0] 
            #reward = self._rewardPred_for_NasimPref(adjustCR)
            print("reward predictor for action", action,  " is = ", reward)
            
            q_values = q_value_batch[i]

            if done:
                q_values[action] = reward
            else:
                #print("q_values before = ", q_values)
                #print future_q_value_batch[i],reward,reward + self.gamma*np.amax(future_q_value_batch[i])
                q_values[action] = reward + self.gamma*np.amax(future_q_value_batch[i])
                #print("q_values after = ", q_values)

            y[i,:] = q_values
            
            x1[i,:] = np.expand_dims(state, axis=0) # use np.epand_dims to make 3D inout to 4D 
            x2[i,:] = np.expand_dims(old_adjustCR, axis=0) ## TODO: correct this to previous action 

        f = open("RL_Qvalue.txt", "a")
        print(q_values, file=f) 
        print("============================================", file=f)
        f.close()
        return [x1,x2],y

    def trim(self):
        self.sample_obs = []
        self.sample_ys = []