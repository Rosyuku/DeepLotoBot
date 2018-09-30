# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 18:05:49 2018

@author: kazuyuki
"""

import twitter
# pip install python-twitter
import config
import scrapeLoto6
import predict

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.cm as cm
#import matplotlib.font_manager
import pandas as pd
import numpy as np
import cv2
import os

if not os.path.exists("media/"):
    os.mkdir("media/")
    
def tweetResultSummary(api):
    
    def createPiechart(data, label, title, figpath):   
        
        ###綺麗に書くためのおまじない###
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size':15})
        
        ###各種パラメータ###
        size=(7,3.5) #凡例を配置する関係でsizeは横長にしておきます。
        col=cm.Spectral(np.arange(len(data))/float(len(data))) #color指定はcolormapから好みのものを。
        
        ###pie###
        plt.figure(figsize=size,dpi=100)
        plt.title(title)
        plt.pie(data,colors=col,counterclock=False,startangle=90,autopct=lambda p:'{:.1f}%'.format(p) if p>=1.0 else '')
        plt.subplots_adjust(left=0,right=0.7)
        plt.legend(label,fancybox=True,loc='center left',bbox_to_anchor=(0.9,0.5))
        plt.axis('equal') 
        plt.savefig(figpath,bbox_inches='tight',pad_inches=0.05)    
    
    logdata = scrapeLoto6.getLogdata(False, False, -1)

    figpath = 'media/figure'+ str(logdata.index[0]).zfill(5) +'.png'
    
    if os.path.exists(figpath) == False:
    
        logdata2 = scrapeLoto6.getLogdata(False, True, -1)
        logdata2['1等賞金総額'] = logdata2['1等口数'] * logdata2['1等賞金']
        logdata2['2等賞金総額'] = logdata2['2等口数'] * logdata2['2等賞金']
        logdata2['3等賞金総額'] = logdata2['3等口数'] * logdata2['3等賞金']
        logdata2['4等賞金総額'] = logdata2['4等口数'] * logdata2['4等賞金']
        logdata2['5等賞金総額'] = logdata2['5等口数'] * logdata2['5等賞金']
        logdata2['ハズレ口数'] = (logdata2.iloc[:, 3] / 200).astype(int) - logdata2['1等口数'] - logdata2['2等口数'] - logdata2['3等口数'] - logdata2['4等口数'] - logdata2['5等口数']
        #logdata2['ハズレ総額'] = logdata2['ハズレ口数'] * 200
        logdata2['運営回収額'] = logdata2.iloc[:, 3] - logdata2['1等賞金総額'] - logdata2['2等賞金総額'] - logdata2['3等賞金総額'] - logdata2['4等賞金総額']- logdata2['5等賞金総額']
        
        logdata['販売口数'] = (logdata2.iloc[:, 3] / 200).astype(int).astype(str) + '口'
        logdata['賞金総額'] = (logdata2['1等賞金総額'] + logdata2['2等賞金総額'] + logdata2['3等賞金総額'] + logdata2['4等賞金総額'] + logdata2['5等賞金総額']).astype(str) + '円'
        logdata['期待値'] = ((logdata2['1等賞金総額'] + logdata2['2等賞金総額'] + logdata2['3等賞金総額'] + logdata2['4等賞金総額'] + logdata2['5等賞金総額']) / (logdata2.iloc[:, 3] / 200)).round().astype(str) + '円'
    
        msg = "%sに実施された%sの結果は、本数字が%s・%s・%s・%s・%s・%s、ボーナス数字が%sでした。また、販売口数は%sで、賞金総額は%sですので、1口あたりの期待値は%sということになります。なお、次回キャリーオーバー額は%sとのことです。 #loto6 #ロト6" % tuple(logdata.iloc[0, [1, 0, 5, 6, 7, 8, 9, 10, 11,  -3, -2, -1, -4]])
        #msg = "%sに実施された%sの結果は、本数字が%s,%s,%s,%s,%s,%s、ボーナス数字が%sとなったようです。なお、販売額としては%sで、次回キャリーオーバー額は%sとのことです。 #loto6 #ロト6" % tuple(logdata.iloc[0, [1, 0, 5, 6, 7, 8, 9, 10, 11, 3, -1]])

        ###データ###
        data=logdata2.iloc[0, [12, 14, 16, 18, 20, -2]].values
        label=['First','Second','Third','Forth','Fifth','Losing']
        title = 'Percentage (entries)'
        fig1path = 'media/figure_entries'+ str(logdata.index[0]).zfill(5) +'.png'
        
        createPiechart(data, label, title, fig1path)
        
        ###データ###
        data=logdata2.iloc[0, [-7, -6, -5, -4, -3, -1]].values
        label=['First','Second','Third','Forth','Fifth',"Host's"]
        title='Percentage (amount)'
        fig2path = 'media/figure_amount'+ str(logdata.index[0]).zfill(5) +'.png'
        
        createPiechart(data, label, title, fig2path)
        
        im1 = cv2.imread(fig1path)
        im2 = cv2.imread(fig2path)
        im_v = cv2.vconcat([im1, im2])
        cv2.imwrite(figpath, im_v)
        
        api.PostUpdate(msg, media=figpath)
        
    else:
        print("tweetResultSummary already exists")    
        
def tweetPredictSummary(api):

    def createBarchart(data, label, title, figpath):
            
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size':10})
        
         ###各種パラメータ###
        size=(12,3.5) #凡例を配置する関係でsizeは横長にしておきます。
        
        ###pie###
        plt.figure(figsize=size,dpi=100)
        plt.title(title)
        plt.ylim(0, 1.0)
        plt.bar(label, data)
        plt.savefig(figpath,bbox_inches='tight',pad_inches=0.05)         
    
    targetNo = scrapeLoto6.getlastNo() + 1
    
    figpath = 'media/predict'+ str(targetNo).zfill(5) +'.png'    
    
    if os.path.exists(figpath) == False:
    
        predict.predictNext()
        
        predict_normal = pd.read_csv("data/predict_normal.csv", index_col=0)
        predict_bonus = pd.read_csv("data/predict_bonus.csv", index_col=0)
        
        normalNo = predict_normal.loc[targetNo].sort_values(ascending=False).index[:6].str.zfill(2).sort_values().tolist()
        bonusNo = predict_bonus.loc[targetNo].sort_values(ascending=False).index[0]
        
        msg = "ディープラーニング（LSTM）を用いた予測によると、第%s回のロト6で選ばれる数字は、本数字が%s・%s・%s・%s・%s・%s、ボーナス数字が%sになる見込みです。ご参考まで。#AI #人工知能 #Deepleaning #ディープラーニング #loto6 #ロト6" % tuple([targetNo] + normalNo + [bonusNo])
        
        data = predict_normal.loc[targetNo].values
        label = predict_normal.columns.tolist()
        title = "Winning number accuracy"
        fig1path = 'media/predict_production'+ str(targetNo).zfill(5) +'.png'
        
        createBarchart(data, label, title, fig1path)
        
        data = predict_bonus.loc[targetNo].values
        label = predict_bonus.columns.tolist()
        title = "Bonus number accuracy"
        fig2path = 'media/predict_bonus'+ str(targetNo).zfill(5) +'.png'
        
        createBarchart(data, label, title, fig2path)
        
        im1 = cv2.imread(fig1path)
        im2 = cv2.imread(fig2path)
        im_v = cv2.vconcat([im1, im2])
        cv2.imwrite(figpath, im_v)
        
        api.PostUpdate(msg, media=figpath)    

    else:
        print("tweetPredictSummary already exists")    

if __name__ == "__main__":

    api = twitter.Api(consumer_key=config.consumer_key,
                      consumer_secret=config.consumer_secret,
                      access_token_key=config.access_token_key,
                      access_token_secret=config.access_token_secret
                      )
    
    tweetResultSummary(api)
    tweetPredictSummary(api)
    
