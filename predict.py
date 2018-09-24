#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 14:31:03 2018

@author: kazuyuki
"""

import updateData
import trainingModel

import pandas as pd
import os

from keras.models import load_model
from glob import glob

def predictNext():
    
    timesteps = trainingModel.timesteps
    
    updateData.update()    
    df = pd.read_csv("data/record.csv", index_col=0).iloc[-(timesteps-1):, 5:12]
    nextNo = df.index.max() + 1

    if os.path.exists("data/predict_normal.csv"):
        predict_normal = pd.read_csv("data/predict_normal.csv", index_col=0)
    else:
        predict_normal = pd.DataFrame(index=range(1, nextNo), columns=range(1, 44), data=0)

    if os.path.exists("data/predict_bonus.csv"):
        predict_bonus = pd.read_csv("data/predict_bonus.csv", index_col=0)
    else:
        predict_bonus = pd.DataFrame(index=range(1, nextNo), columns=range(1, 44), data=0)    
    
    if not predict_normal.index.max() == predict_bonus.index.max() == nextNo:

        modelpath = glob("model/*.h5")[-1]
        model = load_model(modelpath)        
        
        df.loc[nextNo] = 0
        
        onehotdf1 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
        onehotdf2 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
        
        for i in df.index[:-1]:
            print(i)
            tdf = df.loc[i, :]
            onehotdf1.loc[i, tdf.iloc[:6].values] = 1
            onehotdf2.loc[i, tdf.iloc[6]] = 1
            
        nm_x, nm_y = trainingModel.createLSTMdata(onehotdf1.values, timesteps)
        bo_x, bo_y = trainingModel.createLSTMdata(onehotdf2.values, timesteps)
        
        nm_pred, bo_pred = model.predict([nm_x, bo_x])
            
        predict_normal.loc[nextNo] = nm_pred.reshape(-1)
        predict_bonus.loc[nextNo] = bo_pred.reshape(-1)
        
        predict_normal.to_csv("data/predict_normal.csv")
        predict_bonus.to_csv("data/predict_bonus.csv")
        
    else:
        print('predict no update')
        
if __name__ == "__main__":
    
    predictNext()
        
        
