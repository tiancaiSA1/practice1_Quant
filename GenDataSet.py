#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 26 20:24:54 2017

@author: yinboya
"""

import numpy as np
import pandas as pd

class csv2tset:
    def __init__(self, filepath, name, blank = 0):
        self.filepath = filepath + name
        self.Data = pd.read_csv(self.filepath, iterator = True)
        if blank != 0:
            self.Data.get_chunk(blank)
    
    def tData_Regression(self, tNum, FD = 1):
        # FD is an index to enlarge the dependent variable
        tData = self.Data.get_chunk(tNum)
        tData = np.array(tData)
        tX, tY = tData[:,:-1], tData[:,-1] * FD
        return(tX,tY)
    
    def tData_Classification(self, tNum, Class_MaxThreshold, Class_MinThreshold):
        tData = self.Data.get_chunk(tNum)
        tData = np.array(tData)
        tX, tY = tData[:,:-1], tData[:,-1]
        TMaxInd = []
        TMinInd = []
        TMidInd = []
        for i in range(len(tY)):
            if tY[i] > Class_MaxThreshold:
                TMaxInd.append(i)
            elif tY[i] < Class_MinThreshold:
                TMinInd.append(i)
            else:
                TMidInd.append(i)
        tY[TMaxInd] = 1
        tY[TMinInd] = -1
        tY[TMidInd] = 0
        return(tX,tY)