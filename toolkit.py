import os
import collections
import pandas as pd
import numpy as np
from numpy import nan
from IPython.display import display
import math
import matplotlib.pyplot as plt
from itertools import chain

class Toolkit: 
    '''
    load all files in directory and save their names in array
    '''
    def __init__(self): 
        # classification: label prediction
        # regression: quantitiy prediction
        # nominal: 
        # categorical: 
        # listt = list(range(1,17))
        # print('lists: ', listt)
        self.fileNames = list(filter(lambda fileName: 'data' in fileName, os.listdir()))
        self.hasHeaders = ['forestfires.data', ]
        self.datasets = collections.defaultdict()
        self.toStandardize = ['machine.data', 'breast-cancer-wisconsin.data', 'house-votes-84.data', 'forestfires.data', 'abalone.data', 'car.data']
        # self.nominalColumns = {'abalone.data': [0], 'car.data': [5], 'forestfires.data':[2,3], 'house-votes-84.data':[0]}
        # self.ordinalFeatures = {'machine.data': [0,1], 'house-votes-84.data': range(1,17)}
        self.nominalColumns = {}
        self.ordinalFeatures = {'machine.data': [0,1], 'house-votes-84.data': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16], 'abalone.data': [0], 'car.data': [0,1,4,5,6], 'forestfires.data':[2,3]}
        self.continualFeatures = {'abalone.data':[1]}
        self.classificationDatasets = ['breast-cancer-wisconsin.data', 'house-votes-84.data', 'car.data']
        self.regressionDatasets = ['abalone.data', 'forestfires.data', 'machine.data']
        self.columnToPredict = {'abalone.data': 8, 'breast-cancer-wisconsin.data': 10, 'cars.data': 7, 'forestfires.data': 12, 'house-votes-84.data': 0, 'machine.data': 8, 'car.data': 6}
        self.numericalColumns = {'abalone.data': range(1,7), 'breast-cancer-wisconsin.data': range(0,11), 'car.data': [2,3], 'forestfires.data':[0,1,4,5,6,7,8,9,10,11,12], 'house-votes-84.data':[], 'machine.data': range(2,10)}
        self.fileNames = ['car.data']
    '''
    1.1 Loading data
    
    paramters: None
    return: None
    '''
    def load_data(self): 
        print('LOADING DATA')
        for fileName in self.fileNames:
            if fileName == 'breast-cancer-wisconsin.data':
                data = pd.read_csv(fileName, sep=",", header=None, na_values='?')
                self.datasets[fileName] = data
                self.handle_missing_values(fileName)
                # self.handle_catagorical_data(fileName)
            elif fileName == 'forestfires.data':
                data = pd.read_csv(fileName, sep=",", header=None)
                data = data[1:]
            else:
                data = pd.read_csv(fileName, sep=",", header=None)
            
            self.datasets[fileName] = data
            
            self.handle_catagorical_data(fileName)
            self.convert_to_numerics(fileName)
            if fileName == 'car.data':
                self.handle_missing_values(fileName)
        print('dataset after loading: ', self.datasets[fileName].head(5))
        print('loaded data!')
        
    def convert_to_numerics(self, fileName):
        for colNumber in self.numericalColumns[fileName]:
            self.datasets[fileName][colNumber] = self.datasets[fileName][colNumber].apply(pd.to_numeric, errors='coerce')
        
    
    def handle_missing_values(self, fileName):
        print('handle_missing_values: ', fileName)
        display('# NaN before: ', self.datasets[fileName].isnull().sum())
        self.datasets[fileName].fillna(self.datasets[fileName].mean(), inplace=True)
        display('# NaN before: ', self.datasets[fileName].isnull().sum())
        
    '''
    1.3 Handling categorical data    
    Creates integer to value mapping for ordinal features, one hot encodes nomial features
    paramters: fileName
    return: None, modifies the saved datasets inplace
    '''
    def handle_catagorical_data(self, fileName):
        # display('dataset before dropping, dtypes: ', self.datasets[fileName].dtypes)
        featureIndex = self.columnToPredict[fileName]
        
        # handle ordinal features
        if fileName in self.ordinalFeatures: 
            for colNumber in self.ordinalFeatures[fileName]: 
                # create value:int mapping
                uniqueValues = self.datasets[fileName].ix[:,colNumber].unique()
                integerList = range(0, len(uniqueValues))
                mapping = {val:num for val,num in zip(uniqueValues, integerList)}
                self.datasets[fileName].ix[:,colNumber]=self.datasets[fileName].ix[:,colNumber].apply(mapping.get)

        # handle nominal data
        if fileName in self.nominalColumns:
            for colNumber in self.nominalColumns[fileName]: 
                dummyVariables = pd.get_dummies(self.datasets[fileName].iloc[:,colNumber])

                if colNumber == featureIndex: 
                    # make sure to update columnToPredict, since it will no longer be just 1 column
                    self.columnToPredict[fileName] = list(dummyVariables.columns)
                    print('self.columnToPredict[fileName] now: ', self.columnToPredict[fileName])

                sets = [self.datasets[fileName].iloc[:,:-1], dummyVariables, self.datasets[fileName].iloc[:,-1:]]
                self.datasets[fileName] = pd.concat([s.reset_index(drop=True) for s in sets], axis=1)
                
            for colNumber in self.nominalColumns[fileName]: 
                # drop nominal feature afterwards 
                # print('droping col', colNumber)
                # print('column #: ', self.datasets[fileName].columns[colNumber])
                # print('unique vals: ', self.datasets[fileName].iloc[:, colNumber].unique())
                # print('summary: ', self.datasets[fileName][[colNumber]].describe())
                self.datasets[fileName].drop(colNumber, axis=1, inplace=True)

            # display('dataset before drop: ', self.datasets[fileName].head(5))
            # display('dropping these nomial columns: ' , self.nominalColumns[fileName])
            # self.datasets[fileName].drop(self.nominalColumns[fileName], axis=1, inplace=True)
        display('dataset after drop: ', self.datasets[fileName].head(5))
        display('dataset after: ', self.datasets[fileName].head(5))
        

    '''
    1.4 Binning
    Creates equal width binning for continual values
    paramters: fileName
    return: None, modifies the saved datasets inplace
    '''
    def discretization(self, fileName):
        if fileName in self.continualFeatures:
            for featureIndex in self.continualFeatures[fileName]:
                # display('dataset before: ', self.datasets[fileName].head(5))
                bins = pd.cut(self.datasets[fileName][featureIndex],3,labels=['Small', 'Medium', 'Large'])
                self.datasets[fileName][featureIndex] = bins
                # display('dataset after: ', self.datasets[fileName].head(5))
                
        '''
    1.5 Standardization
    Standardizes datasets on z score, uses mean and std from training set
    paramters: train set, test set
    return: None, modifies the saved datasets inplace
    '''
    def standardize(self, train, test):
        print('standardizing')
        display('train before: ', train.head(5))
        # display('test before: ', test.head(5))
        trainOriginal = train.copy()
        # standardize the training set 
        # for colNumber in range(len(train.columns)): 
        for (columnName, columnData) in train.iteritems():
            # don't standardize one hot encoded columns
            if str(columnName).isnumeric():
                featureValues = train.iloc[:,columnName]
                featureValues = pd.to_numeric(featureValues)
                meanF = featureValues.mean()
                if featureValues.sum() > 0:
                    stdF = featureValues.std()
                    standardizedValues = (featureValues-meanF)/stdF
                    # print('standardizedValues: ', standardizedValues)
                    train.iloc[:,columnName] = standardizedValues
  
        # standardsize the test set
        # for colNumber in range(len(test.columns)): 
        for (columnName, columnData) in train.iteritems():
            # don't standardize one hot encoded columns
            if str(columnName).isnumeric():
                featureValues = test.iloc[:,columnName]
                featureValues = pd.to_numeric(featureValues)
                featureValuesTrain = trainOriginal.iloc[:,columnName]
                featureValuesTrain = pd.to_numeric(featureValuesTrain)
                meanF = featureValuesTrain.mean()
                if featureValues.sum() != 0:
                    stdF = featureValuesTrain.std()
                    standardizedValuesTest = (featureValues-meanF)/stdF
                    test.iloc[:,columnName] = standardizedValuesTest
        return train,test
    

    '''
    1.6 Cross validation
    Breaks datset into k folds and tuning/validation set
    paramters: fileName
    return: Folds & validation set, where folds are list of k folds, with each fold containing test and train set
    '''
    def cross_validate(self, fileName, k):
        print('cross validating')
        n = self.datasets[fileName].count()[0]
        rows = self.datasets[fileName].shape[0]
        attributeCount = len(self.datasets[fileName].columns)
        validationIndex = int(0.8*rows)
        isClassification = True if fileName in self.classificationDatasets else False
        folds = []
        
        # copy dataset
        copiedDataset = self.datasets[fileName].copy()

        # shufffle dataset
        shuffledDataset = copiedDataset.iloc[np.random.permutation(len(self.datasets[fileName]))]

        # create the k folds, each time picking a different test set
        leftIndex, rightIndex = 0, int( (1/k) * validationIndex)
        testSetSize = int( (1/k) * validationIndex)
        while rightIndex < validationIndex: 
            testSet = shuffledDataset.iloc[leftIndex:rightIndex]
            trainingSet = pd.concat([shuffledDataset.iloc[:leftIndex], shuffledDataset.iloc[rightIndex+1:]])

            # reset indices for both datasets 
            testSet = testSet.reset_index(drop=True)
            trainingSet = trainingSet.reset_index(drop=True)
            
            folds.append([testSet, trainingSet])
            leftIndex = rightIndex+1
            rightIndex += testSetSize

        # folds contain train/test
        # validate is the rest of the dataset (20%) not in the folds
        validate = shuffledDataset[validationIndex+1:]
        return folds, validate
    
    def evaluateList(self, fileName, trueValues, predictedValues):
        results = {'precision': 0, 'accuracy': 0, 'sumSquared':0}
        columnIndexToPredict = self.columnToPredict[fileName]
        # classification: predict a label, evaluate based on metric 
        if fileName in self.classificationDatasets: 
            TP, TN, FP, FN = 0,0,0,0
            trueValuesBinary = [1 if val == trueValues[0] else 0 for index, val in enumerate(trueValues)]
            predictedValuesBinary = [1 if val == predictedValues[0] else 0 for index, val in enumerate(predictedValues)]

            print('trueValuesBinary: ', trueValuesBinary[1:10])
            print('predictedValuesBinary: ', predictedValuesBinary[1:10])
            for true, predicted in zip(trueValuesBinary, predictedValuesBinary):
                if true == 1 and predicted == 1: 
                    TP += 1
                    continue
                elif true == 0 and predicted == 1: 
                    FP += 1
                    continue
                elif true == 1 and predicted == 0: 
                    FN += 1
                    continue
                elif true == 0 and predicted == 0: 
                    TN += 1
                    continue

            print('TP, TN, FP, FN: ', TP, TN, FP, FN)
            precision = TP / (TP + FP)
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            results['precision'] = precision
            results['accuracy'] = accuracy
                # regression: mean squared error
        else: 
            # convert ture values to numerics as well
            trueValues = pd.DataFrame({'0':trueValues})
            predictedValues = pd.DataFrame({'0':predictedValues})
            print('predictedValues: ', predictedValues)
            # trueValues = trueValues.convert_objects(convert_numeric=True)
            print('trueValues: ', trueValues.head(10))
            print('predictedValues.iloc[:,0]: ', predictedValues.iloc[:,0])
            print('trueValues-predictedValues: ', trueValues-predictedValues)
            print('trueValues.iloc[:,0]-predictedValues.iloc[:,0]', trueValues.iloc[:,0]-predictedValues.iloc[:,0])
            difference = trueValues.iloc[:,0]-predictedValues.iloc[:,0]
            # difference = trueValues-predictedValues.iloc[:,0]
            print('differene: ', difference)
            squared = difference**2
            print('squared: ', squared)
            print('columnIndexToPredict: ', columnIndexToPredict)
            print('squared.sum(): ', squared.sum())
            results['sumSquared'] = np.sqrt(squared.sum())/len(trueValues)
        return results
        
    '''
    1.7 Evaluation
    Calculates precision,accuracy or sum squared based on dataset type
    paramters: fileName, trueVals, predictedVals
    return: dictionary of evaluation metrics, precision, accuracy and sumsquared
    '''
    def evaluate(self, fileName, trueValues, predictedValues):
        print('')
        print('')
        print('-------------EVALUATING-------------------')
        print('trueValues: ', trueValues.head(10))
        trueValues = trueValues.values.tolist()
        trueValues = list(chain.from_iterable(trueValues))
        print('trueValues list: ', trueValues[1:10])
        # print('predictedValues: ', predictedValues.head(10))
        print('predictedValues: ', predictedValues[1:10])
        results = {'precision': 0, 'accuracy': 0, 'sumSquared':0}
        columnIndexToPredict = self.columnToPredict[fileName]
        # classification: predict a label, evaluate based on metric 
        if fileName in self.classificationDatasets: 
            TP, TN, FP, FN = 0,0,0,0
            # convert label into 0/1 values
            # reset column indices so they start from 0
            # trueValues = trueValues.T.reset_index().T.reset_index(drop=True)      
            # predictedValues = predictedValues.T.reset_index().T.reset_index(drop=True)    

            # trueValuesBinary, predictedValuesBinary = [], []
            # firstValTrue = trueValues.values[1:][0]
            # firstPredictedVal = predictedValues.values[1:][0]
            # print('firstPredictedVal: ', firstPredictedVal)
            # print('firstValTrue: ', firstValTrue)

            # for val in trueValues.values[1:]:
            #     print('val[0]', val[0])
            #     print('trueValues.iloc[0][0]: ', trueValues.iloc[0][0])
            #     print('val[0] == trueValues[0][0]: ', val[0] == trueValues[0][0])

            # trueValuesBinary = [1 if val[0] == trueValues[0][0] else 0 for index, val in trueValues.iterrows()]
            # predictedValuesBinary = [1 if val[0] == predictedValues[0][0] else 0 for index, val in predictedValues.iterrows()]

            trueValuesBinary = [1 if val == trueValues[0] else 0 for index, val in enumerate(trueValues)]
            predictedValuesBinary = [1 if val == predictedValues[0] else 0 for index, val in enumerate(predictedValues)]

            print('trueValuesBinary: ', trueValuesBinary[1:10])
            print('predictedValuesBinary: ', predictedValuesBinary[1:10])
            for true, predicted in zip(trueValuesBinary, predictedValuesBinary):
                if true == 1 and predicted == 1: 
                    TP += 1
                    continue
                elif true == 0 and predicted == 1: 
                    FP += 1
                    continue
                elif true == 1 and predicted == 0: 
                    FN += 1
                    continue
                elif true == 0 and predicted == 0: 
                    TN += 1
                    continue

            print('TP, TN, FP, FN: ', TP, TN, FP, FN)
            precision = TP / (TP + FP)
            accuracy = (TP + TN) / (TP + FP + TN + FN)
            results['precision'] = precision
            results['accuracy'] = accuracy

        # regression: mean squared error
        else: 
            # convert ture values to numerics as well
            trueValues = pd.DataFrame({'0':trueValues})
            predictedValues = pd.DataFrame({'0':predictedValues})
            # trueValues = trueValues.convert_objects(convert_numeric=True)
            print('trueValues: ', trueValues.head(10))
            print('predictedValues.iloc[:,0]: ', predictedValues.iloc[:,0])
            print('trueValues-predictedValues: ', trueValues-predictedValues)
            print('trueValues.iloc[:,0]-predictedValues.iloc[:,0]', trueValues.iloc[:,0]-predictedValues.iloc[:,0])
            difference = trueValues.iloc[:,0]-predictedValues.iloc[:,0]
            # difference = trueValues-predictedValues.iloc[:,0]
            print('differene: ', difference)
            squared = difference**2
            print('squared: ', squared)
            print('columnIndexToPredict: ', columnIndexToPredict)
            print('squared.sum(): ', squared.sum())
            results['sumSquared'] = np.sqrt(squared.sum())/len(trueValues)
        return results
    
    
    def learning_algorithms(self, fileName, algorithm, defaultK): 
        averageTestingResult = {'precision': 0, 'accuracy': 0, 'sumSquared':0}
        k = 5
        featureIndex = self.columnToPredict[fileName]
        # 5 fold cross validation, run experiments 5 times
        folds, validation = self.cross_validate(fileName, k)
        
        for idx,fold in enumerate(folds): 
            train, test = fold
            predictedValues = []
            
            # standardize
            if fileName in self.toStandardize: 
                myToolkit.standardize(train, test)
                print('finized standardizing')
            
            if algorithm == 'majority':
                # get algorithm result from training
                predictedValues = self.majority_predictor(fileName, train, test)
            elif algorithm == 'knn':
                predictedValues = self.knn(fileName, train, test)

            # check on validation set (placement for future when comparing multiple algorithms)
            # validationResult = self.evaluate(file, trueValues, predictedValues)

            # get true values from test dataset
            trueValues = pd.DataFrame(train.iloc[:, featureIndex])
            trueValues = trueValues.reset_index(drop=True)

            # evaluate w/ that result with testing set
            evaluationResult = self.evaluate(fileName, trueValues, predictedValues)
            print('evaluationResult for fold #', idx, evaluationResult)
            
            # update overall results
            for metric in averageTestingResult: 
                averageTestingResult[metric] += (evaluationResult[metric]/k)

        print('results for ', fileName)
        print(averageTestingResult)
        return averageTestingResult