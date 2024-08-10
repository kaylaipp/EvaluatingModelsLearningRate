import math
from toolkit import Toolkit
import pandas as pd
import numpy as np
import sys

class LinearNetwork:
    def __init__(self, fileName, train, test):
        self.fileName = fileName
        self.train = train
        self.test = test
        myToolkit = Toolkit()
        myToolkit.fileNames = [fileName]
        myToolkit.load_data()
        self.toolkit = myToolkit
        self.featureToPredict = myToolkit.columnToPredict[fileName]

    '''
    input: array of data
    output: data through logistic sigmoid
    '''
    def sigmoid(self, data):
        return 1 / (1 + np.exp(-data))

    '''
    input: dataframe column
    output: dataframe column converted to 0/1 binary values
    '''
    def convertToBinaryArray(self,dfColumn):
        res = []
        firstVal = dfColumn[0]
        for idx,val in dfColumn.items():
            binary = 0 if val == firstVal else 1
            res.append(binary)
        return np.array(res)


    '''
    MSE
    input: actualValues - list, true values
    input: predictedValues - list, predicted values
    return: float
    '''
    def meanSquaredError(self, actualValues, predictedValues):
        error = 0
        print('actual: ', actualValues[1:5])
        print('predicted: ', predictedValues[1:5])
        for i in range(len(actualValues)):
            error += (actualValues[i] - predictedValues[i]) ** 2 if (actualValues[i] - predictedValues[i]) ** 2 != float('inf') else sys.maxsize
        return error / len(actualValues)

    def getLoss(self, features, target, weights):
        results = np.dot(features, weights)
        n = self.train.shape[1]
        ll = -np.sum( target*results - np.log(1 + np.exp(results)))*(1/n)
        return ll

    '''
    get prediction via equation
    input: features - list, feature list
    input: weights - list, weights for the equation
    input: bias - int, bias constant
    :return: list - list of predictions
    '''
    def getPrediction(self, features, weights, bias):
        return np.dot(features, weights) + bias


    def getAccuracy(self, features, actualLabels, weights):
        data_with_intercept = np.hstack((np.ones((features.shape[0], 0)),
                                 features))
        final_scores = np.dot(data_with_intercept, weights)
        preds = np.round(self.sigmoid(final_scores))
        binaryActualLables = self.convertToBinaryArray(actualLabels)
        accuracy = (preds == binaryActualLables).sum().astype(float) / len(preds)
        return accuracy

    '''
    Logistic regression predicts whether something is T/F instead of predicting something continuous like size
    Can also work with continuous predictions too 

    Goal: Your goal is to find the logistic regression function 𝑝(𝐱) such that the predicted responses 𝑝(𝐱ᵢ) 
    are as close as possible to the actual response 𝑦ᵢ for each observation 𝑖 = 1, …, 𝑛.

    Remember that the actual response can be only 0 or 1 in binary classification problems! 
    This means that each 𝑝(𝐱ᵢ) should be close to either 0 or 1.

    Once you have the logistic regression function 𝑝(𝐱), you can use it to predict the outputs for new and unseen inputs

    Logistic regression determines the best predicted weights 𝑏₀, 𝑏₁, …, 𝑏ᵣ such that the function 𝑝(𝐱) is as close as possible to all actual responses 𝑦ᵢ, 𝑖 = 1, …, 𝑛, where 𝑛 is the number of observations. 
    The process of calculating the best weights using available observations is called model training or fitting.
    '''
    def logisticRegression(self, steps, learningRate, intercept = False):
        features = self.train
        featureToPredict = self.featureToPredict
        actualLabels = features.iloc[:,featureToPredict] if str(featureToPredict).isdigit() else features.loc[:,featureToPredict]
        losses = []
        accuracyPerSteps = []

        if intercept:
            intercept = np.ones((features.shape[0], 1))
            features = np.hstack((intercept, features))
            
        weights = np.zeros(features.shape[1])
        # print('weights before: ', weights)
        
        for step in range(steps):
            # get result
            result = np.dot(features, weights)
            predictions = self.sigmoid(result)

            # update weights via gradient descent
            yhat = featureToPredict - predictions
            gradient = np.dot(features.T, yhat)
            print('gradient: ', gradient)
            print('weight before: ', weights)
            print('adding this to weights: ', learningRate * gradient)
            weights += learningRate * gradient
            print('weight after update: ', weights)
            
            # log losses/accuracy every 100 steps
            if step % 100 == 0:
                loss = self.getLoss(features, featureToPredict, weights)
                print('loss: ', loss)
                losses.append(loss)
                accuracy = self.getAccuracy(features, actualLabels, weights)
                print('accuracy: ', accuracy)
                accuracyPerSteps.append(accuracy)

        accuracy = self.getAccuracy(features, actualLabels, weights)
        return weights, accuracy, losses, accuracyPerSteps


    '''
    multiple linear regression model for regression datasets
    - the regression problems should use mean squared error

    input: X: array, features
    input: y: array, true values
    return: void
    '''
    def linearRegression(self, steps, learningRate, intercept = False):
        # init weights and bias to zero
        weights = np.zeros(self.train.shape[1])
        bias = 0
        features = self.train
        featureToPredict = self.featureToPredict
        actualLabels = features.iloc[:,featureToPredict] if str(featureToPredict).isdigit() else features.loc[:,featureToPredict]
        actualLabels = np.array(actualLabels)
        losses = []
        accuracies = []
        for i in range(steps):
            y_hat = (np.dot(self.train, weights) + bias) + bias

            # log loss and accuracy every 1000 iterations
            if i % 1000 == 0:
                print('iteration: ', i)
                loss = self.meanSquaredError(actualLabels, y_hat)
                losses.append(loss)
                predictedValues = self.getPrediction(features, weights, bias)
                accuracy = self.meanSquaredError(actualLabels, predictedValues)
                accuracy = accuracy if accuracy != float('inf') else sys.float_info.max
                accuracies.append(accuracy)
            
            # calculate the partial derivatives to use to update weights
            partial_w = (1 / self.train.shape[0]) * (2 * np.dot(self.train.T, (y_hat - actualLabels)))
            partial_d = (1 / self.train.shape[0]) * (2 * np.sum(y_hat - actualLabels))

            # make sure theyre not neg infinity
            partial_w = [0 if val in [-float('inf'), float('inf'), np.nan] else val for val in partial_w]
            partial_w = np.array(partial_w)
            partial_d = partial_d if partial_d not in [-float('inf'), float('inf'), np.nan] else 0

            # Update the coefficients for the weights
            weights -= learningRate * partial_w
            bias -= learningRate * partial_d

        predictedValues = self.getPrediction(features, weights, bias)
        accuracy = self.meanSquaredError(actualLabels, predictedValues)
        accuracy = accuracy if accuracy != float('inf') else sys.float_info.max
        print('predictedValues: ', predictedValues[1:5])
        print('actualLabels: ', actualLabels[1:5])
        print('returning this accuracy: ', accuracy)
        return weights, accuracy, losses,accuracies


