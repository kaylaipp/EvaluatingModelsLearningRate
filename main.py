import collections
import os
import random
import sys
import pandas as pd
import numpy as np
from numpy import nan
from IPython.display import display
import matplotlib.pyplot as plt
from itertools import chain

from toolkit import Toolkit
from LinearNetwork import LinearNetwork
from FowardFeedNetwork import FowardFeedNetwork
from Autoencoder import Autoencoder


# load files
myToolkit = Toolkit()
myToolkit.fileNames = ['abalone.data', 'machine.data', 'breast-cancer-wisconsin.data', 'forestfires.data','house-votes-84.data', 'car.data']
myToolkit.load_data()
learningRates = [0.7, 0.5, 0.2, 0.1, 0.05, 0.01, 0.005, 0.0001]
steps = 1000
stepsList = [step for step in range(steps+1) if step > 0 and step % 100 == 0]


'''
Plot single line plot
'''
def plot(fileName, x, y, title):
    plt.plot(x,y)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(fileName + ' ' + title)
    plt.show()

'''
Plot multiple lines on 1 plot
Input:
    mapping - dictionary of file and accuracy lists
    title - title of plot
    typee - type (classsification or regression)
    learningRate - learning rate for these experiements
'''
def plotMultiple(x, mapping, title, typee, learningRate):
    for fileName, y in mapping.items():
        plt.plot(x, y, label=fileName)
    plt.xlabel('Steps')
    plt.ylabel('Accuracy')
    plt.title(title  + ' on ' + typee + ' datasets \n' +  'learning rate = ' + str(learningRate))
    plt.legend(loc="upper left")
    plt.show()



'''
1. A linear network for each of the classification and regression data sets
----------------------------------------------------------------------------
*. Implement logistic regression for the three classification problems
    - The classification problems should use cross-entropy loss
*. Implement a simple linear network for the three regression problems
    - the regression problems should use mean squared error

Input: None
Output: 1 graph depicting accuracy rates for the 3 graphs
'''
def buildNetworkType1():
    yPlotResults = {}
    learningRate = 0.005

    # iterate through classification sets
    for fileName in myToolkit.classificationDatasets:
        # 5 vold cross validation
        results = 0
        print('')
        print('building logistic regression model for : ', fileName)
        averagedLosses = [0 for i in range(steps) if i % 100 == 0]
        averagedLosses = np.array(averagedLosses)
        averagedAccuracyPerSteps = np.array(averagedLosses)
        folds,validate = myToolkit.cross_validate(fileName,5)
        for train,test in folds:
            train,test = myToolkit.standardize(train,test)
            linearNetwork = LinearNetwork(fileName, train, test)
            model, accuracy, loss, accuracyPerSteps = linearNetwork.logisticRegression(steps, learningRate)
            averagedLosses = np.add(averagedLosses, np.array(loss))
            averagedAccuracyPerSteps = np.add(averagedAccuracyPerSteps, np.array(accuracyPerSteps))
            results += accuracy
        # plot(fileName, stepsList, averagedAccuracyPerSteps/3, 'logistic regression')
        yPlotResults[fileName] = averagedAccuracyPerSteps/5
        print('yPlotResults now: ', yPlotResults)
    plotMultiple(stepsList, yPlotResults, 'logistic regression', 'classification', learningRate)

    # iterate through regression sets
    # for fileName in myToolkit.regressionDatasets:
    #     results = 0
    #     # 5 vold cross validation
    #     print('')
    #     print('building multiple linear regression model for : ', fileName)
    #     averagedLosses = [0 for i in range(steps) if i % 1000 == 0]
    #     averagedLosses = np.array(averagedLosses)
    #     averagedAccuracyPerSteps = np.array([0 for i in range(steps) if i % 1000 == 0])
    #     folds,validate = myToolkit.cross_validate(fileName,5)
    #     for train,test in folds:
    #         train,test = myToolkit.standardize(train,test)
    #         linearNetwork = LinearNetwork(fileName, train, test)
    #         model, accuracy, loss, accuracyPerSteps = linearNetwork.linearRegression(steps, learningRate)
    #         averagedLosses = np.add(averagedLosses, np.array(loss))
    #         averagedAccuracyPerSteps = np.add(averagedAccuracyPerSteps, np.array(accuracyPerSteps))
    #         print('model: ', model)
    #         print('accuracy: ', accuracy)
    #         print('averagedLosses now: ', averagedLosses)
    #         print('averagedAccuracyPerSteps now: ', averagedAccuracyPerSteps)
    #         results += accuracy
    #     print('averaged results: ', results/5)
    #     print('plotting for ', fileName)
    #     # plot(fileName, stepsList, averagedLosses)
    #     # yPlotResults[fileName] = averagedLosses/5
    #     yPlotResults[fileName] = averagedAccuracyPerSteps/5
    #     print('yPlotResults now: ', yPlotResults)

    # plotMultiple(stepsList, yPlotResults, 'multiple linear regression', 'regression', learningRate)


# buildNetworkType1()
'''
2. A simple feedforward network with two hidden layers (Input ⇒ Hidden 1 ⇒ Hidden 2 ⇒ Prediction)
for each of the classification and regression data sets
----------------------------------------------------------------------------
- Note that “Prediction” in the above should use a softmax output layer for classification and a linear
output for regression.

Input: None
Output: 1 graph depicting accuracy rates for the 3 graphs
'''
def buildNetworkType2():
    steps = 100
    stepsList = [step for step in range(steps+1) if step > 0 and step % 10 == 0]

    learningRate = 0.005
    yPlotResults = {}

    # iterate through classification sets
    for fileName in myToolkit.classificationDatasets:
        # 5 vold cross validation
        results = 0
        print('')
        print('building 2 layer network for : ', fileName)
        folds,validate = myToolkit.cross_validate(fileName,5)
        averagedLosses = [0 for i in range(steps) if i % 10 == 0]
        averagedLosses = np.array(averagedLosses)
        averagedAccuracyPerSteps = np.array(averagedLosses)
        for train,test in folds:
            train,test = myToolkit.standardize(train,test)
            nn = FowardFeedNetwork(fileName, train, test)
            model, accuracy, loss, accuracyPerSteps = nn.trainNetwork(steps, learningRate)
            # averagedLosses = np.add(averagedLosses, np.array(loss))
            averagedAccuracyPerSteps = np.add(averagedAccuracyPerSteps, np.array(accuracyPerSteps))
            results += accuracy
        yPlotResults[fileName] = averagedAccuracyPerSteps/5
    print('yPlotResults final: ', yPlotResults)
    plotMultiple(stepsList, yPlotResults, 'neural network', 'classification', learningRate)


    # iterate through regression sets
    # for fileName in myToolkit.regressionDatasets:
    #     # 5 vold cross validation
    #     results = 0
    #     print('')
    #     print('building 2 layer network for : ', fileName)
    #     folds,validate = myToolkit.cross_validate(fileName,5)
    #     averagedLosses = [0 for i in range(steps) if i % 10 == 0]
    #     averagedLosses = np.array(averagedLosses)
    #     averagedAccuracyPerSteps = np.array(averagedLosses)
    #     for train,test in folds:
    #         train,test = myToolkit.standardize(train,test)
    #         nn = FowardFeedNetwork(fileName, train, test)
    #         model, accuracy, loss, accuracyPerSteps = nn.trainNetworkRegression(steps, learningRate)
    #         # averagedLosses = np.add(averagedLosses, np.array(loss))
    #         averagedAccuracyPerSteps = np.add(averagedAccuracyPerSteps, np.array(accuracyPerSteps))
    #         print('model: ', model)
    #         print('accuracy: ', accuracy)
    #         print('averagedLosses now: ', averagedLosses)
    #         print('averagedAccuracyPerSteps: ', averagedAccuracyPerSteps)
    #         results += accuracy
    #     print('averaged results: ', results/5)
    #     print('plotting for ', fileName)
    #     print('averagedLoses: ', averagedLosses)
    #     print('averagedLosses/5: ', averagedLosses/5)
    #     yPlotResults[fileName] = averagedAccuracyPerSteps/5
    #     print('yPlotResults now: ', yPlotResults)

    # plotMultiple(stepsList, yPlotResults, 'neural network', 'regression', learningRate)

# buildNetworkType2()
'''
3. A feedforward network where the first hidden layer is trained from an autoencoder and the second
hidden layer is trained from the prediction part of the network (Input ⇒ Encoding ⇒ Hidden ⇒
Prediction) for each of the classification and regression data sets
----------------------------------------------------------------------------
- Note that “Prediction” in the above should use a softmax output layer for classification and a linear
output for regression.

Input: None
Output: 1 graph depicting accuracy rates for the 3 graphs
'''
def buildNetworkType3():
    steps = 300
    stepsList = [step for step in range(steps+1) if step > 0 and step % 10 == 0]

    learningRate = 0.1
    yPlotResults = {}

    # iterate through classification sets
    for fileName in myToolkit.classificationDatasets:
        # 5 vold cross validation
        results = 0
        print('')
        print('building 2 layer auto encoder network for : ', fileName)
        folds,validate = myToolkit.cross_validate(fileName,5)
        averagedLosses = [0 for i in range(steps) if i % 10 == 0]
        averagedLosses = np.array(averagedLosses)
        averagedAccuracyPerSteps = np.array(averagedLosses)
        for train,test in folds:
            train,test = myToolkit.standardize(train,test)
            nn = Autoencoder(fileName, train, test)
            model, accuracy, loss, accuracyPerSteps = nn.trainNetwork(steps, learningRate)
            # averagedLosses = np.add(averagedLosses, np.array(loss))
            averagedAccuracyPerSteps = np.add(averagedAccuracyPerSteps, np.array(accuracyPerSteps))
            results += accuracy
        yPlotResults[fileName] = averagedAccuracyPerSteps/5
    print('yPlotResults final: ', yPlotResults)
    plotMultiple(stepsList, yPlotResults, 'neural network autoencoder layer', 'classification', learningRate)

buildNetworkType3()