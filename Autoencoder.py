import math
from toolkit import Toolkit
import pandas as pd
import numpy as np
import sys

class Autoencoder:
    def __init__(self, fileName, train, test):
        print('fileName: ', fileName)
        self.fileName = fileName
        self.train = train.T
        self.test = test.T
        myToolkit = Toolkit()
        myToolkit.fileNames = [fileName]
        myToolkit.load_data()
        self.toolkit = myToolkit
        self.featureToPredict = myToolkit.columnToPredict[fileName]
        data = np.array(self.train)
        m,n = data.shape
        self.m = m
        self.n = n
        actualLabels = self.train.iloc[self.featureToPredict] if str(self.featureToPredict).isdigit() else self.train.loc[self.featureToPredict]
        data_train=data[0:m].T
        Y_train = data_train[0]
        self.Y_train = Y_train
        self.numberUniquePredictions = len(pd.value_counts(actualLabels))
        self.actualLabels = self.oneHotEncode(actualLabels)
        self.actualLabelsBinary = self.convertToBinaryArray(actualLabels)


    '''
    exp(zi)/sum(exp(zj)) (aka sum over all other classes)
    '''
    def softmax(self, zi):
        res = np.exp(zi) / sum(np.exp(zi))
        return res

    '''
    Get prediction from output layer of neural network
    simply return node with max probablity
    '''
    def getPrediction(self, outputs):
        return np.argmax(outputs, 0)

    '''
    Calcuate accuracy
    input: predictions & actual labels
    output: accuracy - float
    '''
    def getAccuracy(self, predictions, actualLabels):
        return np.sum(predictions == actualLabels) / actualLabels.size

    '''
    cross entropy helper
    input: predictions/actual labels
    output: cross entropy for these labels
    '''
    def crossEntropyHelper(self, predictions, actual):
        return -sum([actual[i]*math.log(predictions[i], 2) for i in range(len(actual))])

    '''
    calculate cross entropy distribution based on prediction/actual
    '''
    def crossEntropy(self, predictions, actualLabels):
        averageCrossEntropy = []
        for i in range(len(actualLabels)):
            # create the distribution for each event {0, 1}
            expected = [1.0 - actualLabels[i], actualLabels[i]]
            predicted = [1.0 - predictions[i], predictions[i]]
            # calculate cross entropy for the two events
            ce = self.crossEntropyHelper(predicted, expected)
            averageCrossEntropy.append(ce)
        return sum(averageCrossEntropy)/len(averageCrossEntropy)

    '''
    relu activation function
    input: feature col
    output: max of the feature col
    '''
    def relu(self, feature):
        return np.maximum(0,feature)

    '''
    input: isClasssification dataset or not - boolean
    output: none, update global output variables
    '''
    def forwardPropagation(self,isClassification):
        # Z1 = output layer 1
        self.Z1 = np.dot(self.W1, self.train) + self.b1         
        self.A1 = self.relu(self.Z1)
        print('Z1: ', self.Z1[1])

        # Z2 = output layer 2
        self.Z2 = np.dot(self.W2, self.Z1) + self.b2
        self.A2 = self.relu(self.Z2)
        print('Z2: ', self.Z2[1])

        # Z3 = final output layer
        self.Z3 = np.dot(self.W3, self.Z2) + self.b3
        self.A3 = self.softmax(self.Z3) if isClassification else self.relu(self.Z3)
        print('Z3: ', self.Z3[1])

    '''
    partial derivative of ReLU activation function 
    used for backpropgation
    '''
    def partialRelu(self, feature):
        return feature>0

    '''
    One hot encode the feature to predict column
    input: feature column Y
    output: one hot encoded feature column Y
    '''
    def oneHotEncode(self, Y):
        Y = Y.astype(int)
        oneHotY = np.zeros((Y.size, int(Y.max()) + 1))
        oneHotY[np.arange(Y.size), Y] = 1
        oneHotY = oneHotY.T
        return oneHotY
    
    '''
    intput: feature column
    output: feature column converted to binary 0/1
    '''
    def convertToBinaryArray(self,dfColumn):
        res = []
        firstVal = dfColumn[0]
        for idx,val in dfColumn.items():
            binary = 0 if val == firstVal else 1
            res.append(binary)
        return np.array(res)

    '''
    Utilize chain rule to caclualte the parital derivatives for each w and b
    with respect to cost this will tell us how much we want to update our weights and bias terms

    Input: none
    output: none, updates global variables
    '''
    def backwardPropagation(self):     
        self.dZ3= self.A3 - self.actualLabels
        print('dZ3: ', self.dZ3[1:5])
        # parital of output
        self.dW3 = (1/self.m) * np.dot(self.dZ3, self.A2.T)
        self.db3 = (1/self.m) * np.sum(self.dZ3, axis=1, keepdims=True)
        print('dW3: ', self.dW3[1:5])

        # parital of hidden layer 2 from output
        self.dZ2 = np.multiply(np.dot(self.W3.T, self.dZ3), self.partialRelu(self.Z2))
        self.dW2 = (1/self.m) * np.dot(self.dZ2, self.A1.T)
        self.db2 = (1/self.m) * np.sum(self.dZ2, axis=1, keepdims=True)

        # parital of hididen layer 1 from hidden layer 2 input
        self.dZ1 = np.multiply(np.dot(self.W2.T, self.dZ2), self.partialRelu(self.Z1))
        self.dW1 = (1/self.m) * np.dot(self.dZ1, self.train.T)
        self.db1 = (1/self.m) * np.sum(self.dZ1, axis=1, keepdims=True)

    def updateWeights(self,learningRate):
        print('weight 1 before: ', self.W1[1:5])
        self.W1 -= learningRate * self.dW1
        print('weight 1 after: ', self.W1[1:5])
        self.b1 -=  learningRate * self.db1    
        self.W2 -= learningRate * self.dW2  
        self.b2 -= learningRate * self.db2   
        self.W3 -= learningRate * self.dW3
        self.b3 -=  learningRate * self.db3    

    '''
    initialize global variable nodes and weights, hidden layer default to size of 10
    W1 = size of weight matrix layer 1
    b1 = size of bias matrix layer 1
    W2 = size of weight matrix layer 2
    b2 = size of bias matrix layer 2
    W3 = size of weight matrix layer 3
    b3 = size of bias matrix layer 3
    '''
    def initNodes(self, isClassification):
        hiddenSize = 10
        outputSize = 10

        # autoencoder layer = 2 layers, but 1st hidden layer in our network
        self.AW1 = np.random.rand(hiddenSize, self.train.shape[0]) - 0.5
        self.Ab1 = np.random.rand(outputSize, 1) - 0.5
        self.AW2 = np.random.rand(10, self.train.shape[0]) - 0.5
        self.Ab2 = np.random.rand(outputSize, 1) - 0.5

        self.W1 = np.random.rand(hiddenSize, self.train.shape[0]) - 0.5
        self.b1 = np.random.rand(hiddenSize, 1) - 0.5
        self.W2 = np.random.rand(outputSize, hiddenSize) - 0.5
        self.b2 = np.random.rand(outputSize, 1) - 0.5
        self.W3 = np.random.rand(self.numberUniquePredictions, outputSize) - 0.5 if isClassification else np.random.rand(1, outputSize) 
        self.b3 = np.random.rand(self.numberUniquePredictions, 1) - 0.5 if isClassification else np.random.rand(1, 1) - 0.5
        self.W1 = self.W1.astype('float128')
        self.b1 = self.b1.astype('float128')
        self.W2 = self.W2.astype('float128')
        self.b2 = self.b2.astype('float128')
        self.W3 = self.W3.astype('float128')
        self.b3 = self.b3.astype('float128')

    '''
    Train 2 layer neural network - classification
    input: number of steps to train, learning rate
    output: weights, overall accuracy of the model, losses, and list of accuracies per steps
    '''
    def trainNetwork(self, steps, learningRate): 
        accuracies, losses = [],[]
        self.initNodes(True)
        actualLabels = self.train.iloc[:,self.featureToPredict] if str(self.featureToPredict).isdigit() else self.train.loc[:,self.featureToPredict]
        for i in range(steps):
            self.forwardPropagation(True)
            self.backwardPropagation()
            self.updateWeights(learningRate)
            if i % 10 == 0:
                print("iteration #: ", i)
                predictions = self.getPrediction(self.A3)
                accuracy = self.getAccuracy(predictions, self.actualLabelsBinary)
                print('accuracy: ', accuracy)
                accuracies.append(accuracy)
        print('accuracies at end: ', accuracies)
        print('accuracy: ', sum(accuracies)/len(accuracies))
        overallAccuracy = sum(accuracies)/len(accuracies)
        return self.W1, overallAccuracy, losses, accuracies

    '''
    Train 2 layer neural network - regression
    input: number of steps to train, learning rate
    output: weights, overall accuracy of the model, losses, and list of accuracies per steps
    '''
    def trainNetworkRegression(self, steps, learningRate): 
        accuracies, losses = [],[]
        self.initNodes(False)
        actualLabels = self.train.iloc[:,self.featureToPredict] if str(self.featureToPredict).isdigit() else self.train.loc[:,self.featureToPredict]
        for i in range(steps):
            self.forwardPropagation(False)
            self.backwardPropagation()
            self.updateWeights(learningRate)
            if i % 100 == 0:
                print("Iteration: ", i)
                predictions = self.getPrediction(A3)
                accuracy = self.getAccuracy(predictions, self.actualLabelsBinary)
                print('accuracy: ', accuracy)
                accuracies.append(accuracy)
        print('accuracies at end: ', accuracies)
        print('accuracies: ', sum(accuracies)/len(accuracies))
        overallAccuracy = sum(accuracies)/len(accuracies)
        return W1, overallAccuracy, losses, accuracies