#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 00:11:14 2018

@author: kazuyuki
"""

import pandas as pd
import scrapeLoto6 as sL
import os

if __name__ == "__main__":
    
    if not os.path.exists("data/"):
        os.mkdir("data/")
    
    if os.path.exists("data/record.csv"):
        df = pd.read_csv("data/record.csv", index_col=0)
        currentLast = df.index.max()
        actualLast = sL.getlastNo()
        
        add_dfs = []
        for i in range(currentLast+1, actualLast+1):
            add_dfs.append(sL.getLogdata(False, True, i))
        add_df = pd.concat(add_dfs)
        newdf = pd.concat([df, add_df])
        
    else:
        newdf = sL.getLogdata(True, True)
        
    newdf.to_csv('data/record.csv')
        