#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 01:04:31 2018

@author: kazuyuki
"""

import twitter
# pip install python-twitter
import config
import updateData

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from keras.models import Model, load_model
from keras.layers.core import Dense, Activation
from keras.layers import Input, BatchNormalization, concatenate
from keras.utils import np_utils, plot_model
from keras.optimizers import Adam

import tensorflow as tf
device_name = tf.test.gpu_device_name()
from keras.layers.recurrent import LSTM
    
if not os.path.exists("model/"):
    os.mkdir("model/")

if not os.path.exists("media/"):
    os.mkdir("media/")
    
timesteps = 20
hidden = 300
Tweet = True

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
    
    nm_output = Dense(data_dim, activation='softmax', name='normal_output')(h)
    bo_output = Dense(data_dim, activation='softmax', name='bonus_output')(h)
    
    model = Model(inputs=[nm_input, bo_input], outputs=[nm_output, bo_output])
    
    model.compile(loss="categorical_crossentropy", optimizer='adam')

    return model 

def createLinechart(data, label, title, figpath):   
    
    ###綺麗に書くためのおまじない###
    plt.style.use('ggplot')
    plt.rcParams.update({'font.size':15})
    
    ###各種パラメータ###
    size=(7,3.5) #凡例を配置する関係でsizeは横長にしておきます。
    
    ###pie###
    plt.figure(figsize=size,dpi=100)
    plt.title(title)
    for i in range(data.shape[1]):
        plt.plot(range(data.shape[0]), data[:, i])
    plt.subplots_adjust(left=0,right=0.7)
    plt.legend(label,fancybox=True)
    plt.axis('equal') 
    plt.savefig(figpath,bbox_inches='tight',pad_inches=0.05)
    
def tweetTrainSummary(msg, figpath):
    
    api = twitter.Api(consumer_key=config.consumer_key,
                      consumer_secret=config.consumer_secret,
                      access_token_key=config.access_token_key,
                      access_token_secret=config.access_token_secret
                      )

    api.PostUpdate(msg, media=figpath)    

if __name__ == "__main__":
    
    updateData.update()
    
    df = pd.read_csv("data/record.csv", index_col=0).iloc[:, 5:12]
    
    onehotdf1 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    onehotdf2 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    
    data_dim = onehotdf1.shape[1]
    
    #1行ごとに実施
    for i in df.index:
        #print(i)
        tdf = df.loc[i, :]
        onehotdf1.loc[i, tdf.iloc[:6].values] = 1
        onehotdf2.loc[i, tdf.iloc[6]] = 1
    
    nm_x, nm_y = createLSTMdata(onehotdf1.values, timesteps)
    bo_x, bo_y = createLSTMdata(onehotdf2.values, timesteps)

    model = createModel(timesteps, data_dim, hidden)
    print(model.summary())
 
    #学習
    history = model.fit([nm_x, bo_x], [nm_y, bo_y],
              batch_size=128,
              epochs=50,
              validation_split=0.1,
              )

    #保存
    model.save("model/trainer" + ".h5")
    
    if Tweet == True:
        
        fig1path = 'media/model_config' + str(df.index.max()).zfill(5) + '.png'
        fig2path = 'media/model_loss' + str(df.index.max()).zfill(5) + '.png'
    
        plot_model(model, to_file=fig1path, show_shapes=True)
    
        data = np.array([history.history['loss'], history.history['val_loss']]).T
        createLinechart(data, ['loss', 'val_loss'],'Model loss', fig2path)
        
        msg = "第%s回までのデータを使って、モデルを再学習しました。学習パラメータとしてはエポック数が%s、ミニバッチサイズが%sで、誤差関数として%sを使用しました。詳細な構成や結果は添付画像をご覧ください。 #AI #人工知能 #Deepleaning #ディープラーニング #loto6 #ロト6" % tuple([df.index.max(), history.params["epochs"], history.params["batch_size"], history.model.loss])

        tweetTrainSummary(msg, [fig1path, fig2path])
