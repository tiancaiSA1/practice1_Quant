#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 14:50:39 2017

@author: yinboya
"""
from __future__ import division

import GAxgboost
from xgboost.sklearn import XGBRegressor
import GenDataSet
import pandas as pd
import numpy as np
I = GAxgboost.I
classMaxThreshold = GAxgboost.classMaxThreshold
classMinThreshold = GAxgboost.classMinThreshold

trainNum = GAxgboost.trainNum # train set number
testNum = GAxgboost.testNum #test set number
regFD = GAxgboost.regFD
blank0 = GAxgboost.blank0

filepath = GAxgboost.filepath
file_output = open("Comp_xgb.txt",'w')

file_output.write("trainNum = %s\n" % trainNum)
file_output.write("testNum = %s\n" % testNum)
file_output.write("blank0 = %s\n" % blank0)

summary = []
for i in range(10):
    name = GAxgboost.list_name[i]
    GAxgboost.fpn(name)
    GAxgboost.changeSample()
    print ("-----%s-----" % name)
    # Read Data
    if I == 1:
        myrf = GenDataSet.csv2tset(filepath = filepath, name = name, blank = blank0)
        trainX, trainY = myrf.tData_Classification(trainNum,
                                                   Class_MaxThreshold = classMaxThreshold,
                                                   Class_MinThreshold = classMinThreshold)
        testX, testY = myrf.tData_Classification(testNum,
                                                 Class_MaxThreshold = classMaxThreshold,
                                                 Class_MinThreshold = classMinThreshold)
    else:
        myrf = GenDataSet.csv2tset(filepath = filepath, name = name, blank = blank0)
        trainX, trainY = myrf.tData_Regression(trainNum, regFD)
        testX, testY = myrf.tData_Regression(testNum, regFD)
    
    
    # extract the best individual
    individual = GAxgboost.main()
    
    """
    xgboost model
    """
    # initialize the model
    xgb_fc = XGBRegressor(learning_rate = individual[0], n_estimators = individual[5],
                       silent = True, objective = "reg:linear",
                       nthread = -1, gamma = 0,
                       min_child_weight = individual[1],max_depth = individual[2],
                       subsample = individual[3],colsample_bylevel = individual[4],
                       seed = 0)
    # fit the model by training set
    xgb_fc.fit(trainX, trainY)
    
    # predict training set and calculate the mse
    
    print ("-----xgboost regression-----")
    trainY_pre = xgb_fc.predict(trainX)
    xgb_train_MSE = sum((trainY - xgb_fc.predict(trainX)) ** 2) / len(trainY)
    print ("training set prediction MSE = %s " % xgb_train_MSE)
    # predict testing set and calculate the mse
    testY_pre = xgb_fc.predict(testX)
    xgb_test_MSE = sum((testY - xgb_fc.predict(testX)) ** 2) / len(testY)
    print ("testing set prediction MSE = %s " % xgb_test_MSE)
    
    
    file_output.write("-----%s-----\n" % name)
    file_output.write("-----xgboost regression-----\n")
    file_output.write("training set prediction MSE = %s\n" % xgb_train_MSE)
    file_output.write("testing set prediction MSE = %s\n" % xgb_test_MSE)
    """
    linear model
    """
    from sklearn import linear_model
    linear_fc = linear_model.LinearRegression(n_jobs = -1)
    linear_fc.fit(trainX, trainY)
    trainY_pre = linear_fc.predict(trainX)
    linear_train_MSE = sum((trainY - linear_fc.predict(trainX)) ** 2) / len(trainY)
    print ("-----linear regression-----")
    print ("training set prediction MSE = %s" % linear_train_MSE)
    # predict testing set and calculate the mse
    testY_pre = linear_fc.predict(testX)
    linear_test_MSE = sum((testY - linear_fc.predict(testX)) ** 2) / len(testY)
    print ("testing set prediction MSE = %s" % linear_test_MSE)
    
    varTestY = np.var(testY)
    print ("testing set var = %s" % varTestY)
    
    file_output.write("-----linear regression-----\n")
    file_output.write("training set prediction MSE = %s\n" % linear_train_MSE)
    file_output.write("testing set prediction MSE = %s\n" % linear_test_MSE)
    file_output.write("\n")
    file_output.write("testing set prediction MSE = %s\n" % varTestY)
    file_output.write("\n")
    
    i_summary = [name, xgb_train_MSE, xgb_test_MSE, 
                 linear_train_MSE, linear_test_MSE, varTestY]
    summary.append(i_summary)
summary = pd.DataFrame(summary,
                       columns = ["stockName", "xgb_train_MSE", "xgb_test_MSE", 
                       "linear_train_MSE", "linear_test_MSE", "var_testY"])
summary.to_csv(filepath + "summary.csv")
file_output.close()


