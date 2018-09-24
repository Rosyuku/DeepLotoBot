#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 01:04:31 2018

@author: kazuyuki
"""

import updateData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Model, load_model
from keras.layers.core import Dense, Activation
from keras.layers import Input, BatchNormalization, concatenate
from keras.utils import np_utils
from keras.optimizers import Adam

import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name == '':
    from keras.layers.recurrent import LSTM
else:
    from keras.layers import CuDNNLSTM as LSTM
    
if not os.path.exists("model/"):
    os.mkdir("model/")
    
timesteps = 10
hidden = 100

def createLSTMdata(data, timesteps=10):
    
   data_dim = data.shape[1]
 
   lstm_data = []
   index_data = []
 
   for i in range(timesteps):
 
       length = data[i:].shape[0] // timesteps
       lstm_data.append(data[i:i+length*timesteps].reshape(length, timesteps, data_dim))
       index_data.append(np.arange(i, i+(length*timesteps), timesteps))
 
   lstm_data = np.concatenate(lstm_data, axis=0)
   index_data = np.concatenate(index_data, axis=0)
   lstm_data = lstm_data[pd.Series(index_data).sort_values().index]
 
   lstm_data_x = lstm_data[:, :-1, :]
   lstm_data_y = lstm_data[:, -1, :]
   
   return lstm_data_x, lstm_data_y

def createModel(timesteps, data_dim, hidden=100):
    #モデル定義
    nm_input = Input(shape=(timesteps-1, data_dim), name='normal_input')
    bo_input = Input(shape=(timesteps-1, data_dim), name='bonus_input')
    
    h1 = LSTM(hidden, input_shape=(timesteps-1, data_dim), stateful=False, return_sequences=True)(nm_input)
    h2 = LSTM(hidden, input_shape=(timesteps-1, data_dim), stateful=False, return_sequences=True)(bo_input)
    
    h1 = LSTM(hidden, input_shape=(timesteps-1, data_dim), stateful=False, return_sequences=False)(h1)
    h2 = LSTM(hidden, input_shape=(timesteps-1, data_dim), stateful=False, return_sequences=False)(h2)
    
    h = concatenate([h1, h2])
    
    nm_output = Dense(data_dim, activation='softmax')(h)
    bo_output = Dense(data_dim, activation='softmax')(h)
    
    model = Model(inputs=[nm_input, bo_input], outputs=[nm_output, bo_output])
    
    model.compile(loss="categorical_crossentropy", optimizer='adam')

    return model    

if __name__ == "__main__":
    
    updateData.update()
    
    df = pd.read_csv("data/record.csv", index_col=0).iloc[:, 5:12]
    
    onehotdf1 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    onehotdf2 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    
    data_dim = onehotdf1.shape[1]
    
    #1行ごとに実施
    for i in df.index:
        print(i)
        tdf = df.loc[i, :]
        onehotdf1.loc[i, tdf.iloc[:6].values] = 1
        onehotdf2.loc[i, tdf.iloc[6]] = 1
    
    nm_x, nm_y = createLSTMdata(onehotdf1.values, timesteps)
    bo_x, bo_y = createLSTMdata(onehotdf2.values, timesteps)

    model = createModel(timesteps, data_dim, hidden)
    print(model.summary())
 
    #学習
    history = model.fit([nm_x, bo_x], [nm_y, bo_y],
              batch_size=32,
              epochs=20,
              validation_split=0.1,
              )
    
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])

    #保存
    model.save("model/trainer_" + str(df.index.max()).zfill(5) + ".h5")