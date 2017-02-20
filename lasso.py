#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 17:56:46 2017

@author: yinboya
"""

from __future__ import division
import numpy as np
import GenDataSet
import os
from sklearn import linear_model


popNum = 10 # Population
N_splits = 2 # Crossvalidation fold number
iter_NGEN = 5 # Generation number
iter_CXPB = 0.8 # Crossover probability
iter_MUTPB = 0.9 # Mutation probability
classMaxThreshold = 0.001
classMinThreshold = -0.001

regFD = 10000 # regFD is an index to enlarge the dependent variable for regression

trainNum = 25000 # train set number
testNum = 500 # test set number
blank0 = 0

'''
    This version add parameters
'''
window = 300 # length of window
N_validation = 3 # rolling windows test number when GA evaluation
MUTPB_tN = 0.5 # the probablity of training set number





'''
train set = dataset[range(trainNum)+blank,:]
test set = dataset[range(trainNum)+blank+trainNum,:]
'''

filepath = '/Users/yinboya/STUDY/QuantitativeInvestment/practice1/outdata/'
list_name = os.listdir(filepath)

name = "sym_1.csv"
def fpn(fname):
    global name
    name = fname



# Read Data
myrf = GenDataSet.csv2tset(filepath = filepath, name = name, blank = blank0)
trainX, trainY = myrf.tData_Regression(trainNum, regFD)
testX, testY = myrf.tData_Regression(testNum, regFD)

def changeSample():
    global trainX,trainY,testX,testY
    myrf = GenDataSet.csv2tset(filepath = filepath, name = name,blank = blank0)
    trainX, trainY = myrf.tData_Regression(trainNum, regFD)
    testX, testY = myrf.tData_Regression(testNum, regFD)

# Setting Things Up
import random
from deap import base
from deap import creator
from deap import tools

# Creator
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox = base.Toolbox()


def linear_alpha(MAXalpha = 20, MINalpha = 0.01):
    if random.random() <= 0.5:
        return((1 - MINalpha)*random.random() + MINalpha)
    else:
        return((MAXalpha - 1)*random.random() + 1)


def trainsetNum(MAXnm = 24000, MINnm = 24000):
    return(random.randint(MINnm, MAXnm))

# register xgb parameters
toolbox.register("attr_linear_alpha", linear_alpha)
toolbox.register("attr_trainsetNum",trainsetNum)


# Structure initializers
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_linear_alpha,
                  toolbox.attr_trainsetNum), 1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# The Evaluation Function

def lasso_reg_evaluation(individual):
    trainNumber = individual[1] # the train num
    roll_win_mseValue = 0
    for i in xrange(N_validation):
        trainingX, trainingY = trainX[(trainNum - (i + 1) * window - trainNumber):(trainNum - (i + 1) * window),:],\
                                      trainY[(trainNum - (i + 1) * window - trainNumber):(trainNum - (i + 1) * window)]
                                      
        testingX, testingY= trainX[(trainNum - (i + 1) * window):(trainNum - i * window),:], \
                                   trainY[(trainNum - (i + 1) * window):(trainNum - i * window)]
                                   
        reg = linear_model.Lasso(alpha = individual[0])
        reg.fit(trainingX,trainingY)
        roll_win_mseValue += sum((testingY - reg.predict(testingX)) ** 2) / window
    roll_win_mseValue /= N_validation
    return(roll_win_mseValue,)


# The mutate Function
def lasso_mutation(individual):
    N = len(individual) - 1
    rn = random.randint(1,N)
    if rn == 1 :
        individual[0] = toolbox.attr_linear_alpha()
    

def lasso_trainsetNum_mutation(individual):
    individual[1] = toolbox.attr_trainsetNum()
    return(individual)

# The Genetic Operators
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", lasso_mutation)
toolbox.register("select", tools.selTournament, tournsize = 3)
toolbox.register("mutate_tNum", lasso_trainsetNum_mutation)

toolbox.register("evaluate", lasso_reg_evaluation)

def main(NGEN = iter_NGEN, CXPB = iter_CXPB, MUTPB = iter_MUTPB):
    # Creating the Population
    pop = toolbox.population(n=popNum)
    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
    # Begin the evolution
    for g in range(NGEN):
        print("-- Generation %i --" % g)
        # Select the next generation individuals
        offspring = toolbox.select(pop, len(pop))
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, offspring))
        # Apply crossover and mutation on the offspring
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < CXPB:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values
        
        
        for mutant in offspring:
            if random.random() < MUTPB:
                toolbox.mutate(mutant)
                del mutant.fitness.values
            if random.random() < MUTPB_tN:
                toolbox.mutate_tNum(mutant)
        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
        pop[:] = offspring
        
        # Gather all the fitnesses in one list and print the stats
        fits = [ind.fitness.values[0] for ind in pop]
        
        length = len(pop)
        mean = sum(fits) / length
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / length - mean**2)**0.5
        
        print("  Min %s" % min(fits))
        print("  Max %s" % max(fits))
        print("  Avg %s" % mean)
        print("  Std %s" % std)
        print("  Parameter %s" % offspring[np.argmin(fits)])
    return(offspring[np.argmin(fits)])


if __name__ == '__main__':
    main()
