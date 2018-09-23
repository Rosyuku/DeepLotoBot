# -*- coding: utf-8 -*-
"""
Created on Sun Jul 30 21:50:59 2017

@author: Wakasugi Kazuyuki
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup

def getlastNo():
    url = "http://sougaku.com/loto6/data/detail/"
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")
    last = int(soup.find(class_='contents_ti6').find_all('h3')[0].text[4:8])
    return last    
    
def getcolumnsList():
    columnsList=[
     '抽せん回',
     '抽せん日',
     '曜日',     
     '販売実績額',
     'セット球',
     '本数字1',
     '本数字2',
     '本数字3',
     '本数字4',
     '本数字5',
     '本数字6',
     'ボーナス数字',
     '1等口数',
     '1等賞金',
     '2等口数',
     '2等賞金',
     '3等口数',
     '3等賞金',
     '4等口数',
     '4等賞金',
     '5等口数',
     '5等賞金',
     'キャリーオーバー',
     ]
    return columnsList

def getdataList(url="http://sougaku.com/loto6/data/detail/"):
    dataList = []
    res = requests.get(url)
    soup = BeautifulSoup(res.content, "html.parser")
    tb1 = soup.find(class_='sokuho_tb1')
    tb2 = soup.find(class_='sokuho_tb2')
    tb3 = soup.find(class_='sokuho_tb3')    
    for i in range(len(tb1.find_all('td'))):
        dataList.append(tb1.find_all('td')[i].text.replace('\n', '').replace('\t', ''))    
    for i in range(len(tb2.find_all('td'))):
        dataList.append(tb2.find_all('td')[i].text.replace('\n', '').replace('\t', ''))
    for i in range(len(tb3.find_all('td'))):
        dataList.append(tb3.find_all('td')[i].text.replace('\n', '').replace('\t', ''))    
    return pd.Series(dataList)[[0, 1, 1, 2, 3, 12, 13, 14, 15, 16, 17, 19, 21, 22, 25, 26, 29, 30, 33, 34, 37, 38, 41]]

def getLogdata(getall=True, modify=True, last=-1):
    
    if last == -1:
        last = getlastNo()
    
    columnsList = getcolumnsList()   

    if getall == True: 
        logdata = pd.DataFrame(index=range(1, last+1), columns=columnsList)
    else:
        logdata = pd.DataFrame(index=range(last, last+1), columns=columnsList)

    for i in logdata.index:
        
        target = i
        print(i)
        
        try:
            url = "http://sougaku.com/loto6/data/detail/index" + str(target) + ".html"
            logdata.loc[i, :] = getdataList(url).values
        except:
            url = "http://sougaku.com/loto6/data/detail/"
            logdata.loc[i, :] = getdataList(url).values
            
#        if target == last:
#            url = "http://sougaku.com/loto6/data/detail/"
#        else:
#            url = "http://sougaku.com/loto6/data/detail/index" + str(target) + ".html"
#        
#        logdata.loc[i, :] = getdataList(url).values

    if modify == False:
        return logdata
    
    pattern = '|'.join(["第", "回", "円", "口", ",", "該当なし"])

    for i in range(logdata.columns.shape[0]):
        
        if i == 0 or i == 3 or i >= 5:
            logdata.iloc[:, i] = pd.to_numeric(logdata.iloc[:, i].str.replace(pattern, "")).fillna(0).astype(int)
        elif i == 1:
            logdata.iloc[:, i] = logdata.iloc[:, i].str[:-3]
        elif i == 2:
            logdata.iloc[:, i] = logdata.iloc[:, i].str[-2]
            
    return logdata

if __name__ == "__main__":
    
    logdata = getLogdata(True, True)
    logdata.to_csv('data/record.csv')