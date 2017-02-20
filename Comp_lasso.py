#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Feb 20 11:19:50 2017

@author: yinboya
"""

from __future__ import division
from sklearn import linear_model
import lasso
import GenDataSet
import pandas as pd
import numpy as np

trainNum = lasso.trainNum # train set number
testNum = lasso.testNum #test set number
regFD = lasso.regFD
blank0 = lasso.blank0

window = lasso.window


filepath = lasso.filepath
file_output = open("Comp_lasso.txt",'w')

file_output.write("trainNum = %s\n" % trainNum)
file_output.write("testNum = %s\n" % testNum)
file_output.write("blank0 = %s\n" % blank0)

summary = []
for i in range(10):
    name = lasso.list_name[i]
    lasso.fpn(name)
    try:
        lasso.changeSample()
    except Exception:
        file_output.write("There is something wrong in %s\n" % name)
        print("There is something wrong in %s" % name)
        continue
    print ("-----%s-----" % name)
    # Read Data
    
    myrf = GenDataSet.csv2tset(filepath = filepath, name = name, blank = blank0)
    trainX, trainY = myrf.tData_Regression(trainNum, regFD)
    testX, testY = myrf.tData_Regression(testNum, regFD)
    
    
    # extract the best individual
    individual = lasso.main()
    
    """
    xgboost model
    """
    # initialize the model     
    trainNumber = individual[1] # the train num
    
    trainingX, trainingY = trainX[(trainNum - trainNumber):(trainNum),:],\
                                  trainY[(trainNum - trainNumber):(trainNum)]
    
    trainNumber = individual[1] # the train num       
    #reg = linear_model.Lasso(alpha = individual[0])
    reg = linear_model.Lasso(alpha = 0)
    reg.fit(trainingX,trainingY)
    
    # predict training set and calculate the mse
    print ("-----lasso regression-----")
    #trainY_pre = xgb_fc.predict(trainingX)
    lasso_train_MSE = sum((trainingY - reg.predict(trainingX)) ** 2) / len(trainY)
    print ("training set prediction MSE = %s " % lasso_train_MSE)
    # predict testing set and calculate the mse
    #testY_pre = xgb_fc.predict(testingX)
    lasso_test_MSE = sum((testY - reg.predict(testX)) ** 2) / len(testY)
    print ("testing set prediction MSE = %s " % lasso_test_MSE)
    
    
    file_output.write("-----%s-----\n" % name)
    file_output.write("-----lasso regression-----\n")
    file_output.write("training set prediction MSE = %s\n" % lasso_train_MSE)
    file_output.write("testing set prediction MSE = %s\n" % lasso_test_MSE)

    """
    linear model
    """
    linear_fc = linear_model.LinearRegression(n_jobs = -1)
    linear_fc.fit(trainX, trainY)
    trainY_pre = linear_fc.predict(trainX)
    linear_train_MSE = sum((trainY - linear_fc.predict(trainX)) ** 2) / len(trainY)
    linear_train_R2 = 1 - (sum((trainY - linear_fc.predict(trainX)) ** 2) / \
                           sum((trainY - np.mean(trainY)) ** 2))
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
    file_output.write("testing set Y's var = %s\n" % varTestY)
    file_output.write("\n")
    
    i_summary = [name, lasso_train_MSE, lasso_test_MSE, 
                 linear_train_MSE, linear_test_MSE,
                 linear_train_R2, varTestY, individual]
    summary.append(i_summary)
summary = pd.DataFrame(summary,
                       columns = ["stockName", "lasso_train_MSE", "lasso_test_MSE", 
                       "linear_train_MSE", "linear_test_MSE", "linear_train_R2",
                       "var_testY", "param"])
summary.to_csv("summary.csv")
file_output.close()


