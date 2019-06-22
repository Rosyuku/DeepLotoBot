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
from scipy.special import comb

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
    
        msg = "%sに実施された%sの結果は、本数字が%s・%s・%s・%s・%s・%s、ボーナス数字が%sでした。また、販売口数は%sで、賞金総額は%sですので、1口あたりの期待値は%sです。なお、次回キャリーオーバー額は%sです。 #loto6 #ロト6" % tuple(logdata.iloc[0, [1, 0, 5, 6, 7, 8, 9, 10, 11,  -3, -2, -1, -4]])
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
        
        api.PostUpdate(msg.replace(",", ""), media=figpath)
        
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
        
        msg = "ディープラーニング（LSTM）を用いた予測によると、第%s回のロト6で選ばれる数字は、本数字が%s・%s・%s・%s・%s・%s、ボーナス数字が%sになる見込みです。ご参考まで。#AI #人工知能 #Deeplearning #ディープラーニング #loto6 #ロト6" % tuple([targetNo] + normalNo + [bonusNo])
        
        data = predict_normal.loc[targetNo].values
        label = predict_normal.columns.tolist()
        title = "Winning number softmax score"
        fig1path = 'media/predict_production'+ str(targetNo).zfill(5) +'.png'
        
        createBarchart(data, label, title, fig1path)
        
        data = predict_bonus.loc[targetNo].values
        label = predict_bonus.columns.tolist()
        title = "Bonus number softmax score"
        fig2path = 'media/predict_bonus'+ str(targetNo).zfill(5) +'.png'
        
        createBarchart(data, label, title, fig2path)
        
        im1 = cv2.imread(fig1path)
        im2 = cv2.imread(fig2path)
        im_v = cv2.vconcat([im1, im2])
        cv2.imwrite(figpath, im_v)
        
        api.PostUpdate(msg, media=figpath)    

    else:
        print("tweetPredictSummary already exists")
        
def tweetValidationSummary(api):

    def plotbar(size, title, df_data, figpath):
        plt.style.use('ggplot')
        plt.rcParams.update({'font.size':10})    
        
        plt.figure(figsize=size, dpi=100)
        df_data.plot.bar(title=title, figsize=size)
        plt.xlabel("Number o hits")
        plt.ylabel("Frequency")
        plt.savefig(figpath, bbox_inches='tight', pad_inches=0.05)      
    
    #正解データを読み出し
    df = pd.read_csv('data/record.csv', index_col=0).iloc[:, 5:12]
    onehotdf1 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    onehotdf2 = pd.DataFrame(index=df.index, columns=range(1, 44), data=0)
    ids1 = pd.melt(df.iloc[:, :6].reset_index(), id_vars='index')
    ids2 = pd.melt(df.iloc[:, [6]].reset_index(), id_vars='index')
    for i in range(1, 44):
        onehotdf1.loc[ids1.loc[ids1['value']==i, 'index'].values, i] = 1
        onehotdf2.loc[ids2.loc[ids2['value']==i, 'index'].values, i] = 1
    
    #予測結果を読み出し
    predict_normal = pd.read_csv("data/predict_normal.csv", index_col=0)
    predict_bonus = pd.read_csv("data/predict_bonus.csv", index_col=0)
    predict_normal = predict_normal[predict_normal.sum(axis=1) != 0]
    predict_bonus = predict_bonus.loc[predict_normal.index]
    
    #答え合わせデータを作成
    df_result = pd.DataFrame(index=predict_normal.index, columns=['correctnNum', 'correctbNum'], data=np.NaN)
    for targetNo in predict_normal.index:
        try:
            normalNo = predict_normal.loc[targetNo].sort_values(ascending=False).index[:6]
            bonusNo = predict_bonus.loc[targetNo].sort_values(ascending=False).index[0]
            correctnNum = onehotdf1.loc[targetNo, normalNo.astype(int)].sum()
            correctbNum = onehotdf2.loc[targetNo, int(bonusNo)].sum()
            df_result.loc[targetNo] = correctnNum, correctbNum
        except:
            pass
    df_result = df_result.loc[df_result.count(axis=1) == 2]
    
    #理論値との比較結果を作成
    df_tmp1 = df_result['correctnNum'].value_counts()
    df_tmp2 = df_result['correctbNum'].value_counts()
    df_compare1 = pd.DataFrame(index=range(0, 7), columns=['result(LSTM)', 'theory(random)'], data=0)
    df_compare1.loc[df_tmp1.index, 'result(LSTM)'] = df_tmp1
    df_compare2 = pd.DataFrame(index=range(0, 2), columns=['result(LSTM)', 'theory(random)'], data=0)
    df_compare2.loc[df_tmp2.index, 'result(LSTM)'] = df_tmp2
    
    for i in df_compare1.index:
        df_compare1.loc[i, 'theory(random)'] = comb(37, (6-i)) * comb(6, i) / comb(43, 6) * df_result.shape[0]
    for i in df_compare2.index:
        df_compare2.loc[i, 'theory(random)'] = comb(43, (1-i)) * comb(1, i) / comb(43, 1) * df_result.shape[0]
    
    #既にツイート済みかのチェック
    targetNo = df_result.index[-1]
    figpath = 'media/validate'+ str(targetNo).zfill(5) +'.png'
    
    if os.path.exists(figpath) == False:
        #グラフ作成
        fig1path = 'media/validate_winning'+ str(targetNo).zfill(5) +'.png'
        fig2path = 'media/validate_bonus'+ str(targetNo).zfill(5) +'.png'    
        plotbar((7,3.5), "Hit times of Winning number", df_compare1.iloc[:], fig1path)
        plotbar((4,3.5), "Hit times of Bonus number", df_compare2.iloc[:], fig2path)    
            
        #ツイート文章作成
        normalNo = predict_normal.loc[targetNo].sort_values(ascending=False).index[:6]
        bonusNo = predict_bonus.loc[targetNo].sort_values(ascending=False).index[0]
        correctnNum = onehotdf1.loc[targetNo, normalNo.astype(int)].sum()
        correctbNum = onehotdf2.loc[targetNo, int(bonusNo)].sum()
        correctnNumber = onehotdf1.loc[targetNo, normalNo.astype(int)].loc[onehotdf1.loc[targetNo, normalNo.astype(int)] > 0].index.tolist()
        correctbNumber = onehotdf1.loc[targetNo, [int(bonusNo)]].loc[onehotdf2.loc[targetNo, [int(bonusNo)]] > 0].index.tolist()
        predNum = df_result.shape[0]
        averageacc = np.round((df_compare1.multiply(df_compare1.index, axis=0).sum() / predNum).loc['result(LSTM)'], 2)
        averageaccdiff = np.round(averageacc - (df_compare1.multiply(df_compare1.index, axis=0).sum() / predNum).loc['theory(random)'], 2)
        msg = "第%s回ロト6の予測結果は、本数字が%s個（%s）ボーナス数字が%s個（%s）の的中でした。過去%s回の予測では本数字を平均%s個的中させ、ランダム予測の期待値（0.84個）と比べて%s個の差が出ています（内訳はグラフ参照）。#AI #人工知能 #Deeplearning #ディープラーニング #loto6 #ロト6" % tuple([targetNo] + [correctnNum] + [correctnNumber]  + [correctbNum] + [correctbNumber] + [predNum] + [averageacc] + [averageaccdiff])
        
        #ツイート画像作成
        im1 = cv2.imread(fig1path)
        im2 = cv2.imread(fig2path)
        im_v = cv2.hconcat([im1, im2])
        cv2.imwrite(figpath, im_v)
        
        #ツイート実行
        api.PostUpdate(msg, media=figpath)    

    else:
        print("tweetValidateSummary already exists")
    
if __name__ == "__main__":

    api = twitter.Api(consumer_key=config.consumer_key,
                      consumer_secret=config.consumer_secret,
                      access_token_key=config.access_token_key,
                      access_token_secret=config.access_token_secret
                      )
    
    tweetResultSummary(api)
    tweetPredictSummary(api)
    tweetValidationSummary(api)
    
