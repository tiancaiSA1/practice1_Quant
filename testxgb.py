#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 14:51:38 2017

@author: yinboya
"""

'''
test
'''

import pandas as pd
import numpy as np
import xgboost as xgb
import time
filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice1/outdata/'
Data = pd.read_csv(filepath + 'sym_1.csv')
Data = np.array(Data)
FD = 1000
trainingX, trainingY = Data[:,:-1], Data[:,-1] * FD
individual = [0.02, 20, 8, 0.7, 0.7, 100]
param = {'eta' : individual[0],
         'silent' : True, 'objective' : "reg:linear", 'nthread' : -1,
         'min_child_weight' : individual[1],'max_depth' : individual[2],
         'subsample' : individual[3], 'colsample_bylevel' : individual[4],
         'seed' : 0}
dtrain = xgb.DMatrix(data= trainingX, label = trainingY)


for i in range(10):
    bst = xgb.Booster({'nthread':i})
    t0 = time.time()
    bst = xgb.train(params = param, dtrain = dtrain, num_boost_round = individual[5])
    t1 = time.time()
    print "Thread%s : total time is %s second" % (i,t1 - t0)

