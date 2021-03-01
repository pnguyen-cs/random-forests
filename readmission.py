#Code framework courtesy of Jason Brownlee
#https://machinelearningmastery.com/naive-bayes-classifier-scratch-python/

import csv
import random
import math
import numpy as np

#Load data from csv file
def loadCsv(filename):
    lines = csv.reader(open(filename, newline=''), delimiter=',', quotechar='|')
    next(lines)
    dataset = []
    for row in lines:
        dataset.append([float(x) for x in row])
    return np.array(dataset, dtype=np.float)

#Split data into training and validation portions
def splitDataset(dataset, splitRatio):
    trainSize = int(len(dataset) * splitRatio)
    copy = np.copy(dataset)
    np.random.shuffle(copy)
    trainSet = copy[:, :trainSize]
    copy = copy[:, trainSize:]
    return [trainSet, copy]

#Group data points by class
def separateByClass(dataset):
    separated = {}
    for i in range(len(dataset)):
        vector = dataset[i]
        if (vector[-1] not in separated):
            separated[vector[-1]] = []
        separated[vector[-1]].append(vector)
    return separated

#Compute the mean across a collection of numbers
def mean(numbers):
    return sum(numbers)/float(len(numbers))

#Compute the standard deviation across a collection of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([pow(x-avg,2) for x in numbers])/float(len(numbers)-1)
    return math.sqrt(variance)

#Compute summary statistics for the entire dataset
def summarize(dataset):
    summaries = [(mean(attribute), stdev(attribute)) for attribute in zip(*dataset)]
    del summaries[-1]
    return summaries

#Compute summary statistics per class
def summarizeByClass(dataset):
    separated = separateByClass(dataset)
    summaries = {}
    print (separated)
    for classValue, instances in separated.items():
        summaries[classValue] = summarize(instances)
    return summaries

#Compute per-attribute probabilities
def calculateProbability(x, mean, stdev):
    if stdev == 0:
        return 0.000001
    exponent = math.exp(-(math.pow(x-mean,2)/(2*math.pow(stdev,2))))
    return max(0.000001,(1 / (math.sqrt(2*math.pi) * stdev)) * exponent)

#Calculate global calculateProbability
#To prevent data type underflow, we use log probabilities
def calculateClassProbabilities(summaries, inputVector):
    probabilities = {}
    for classValue, classSummaries in summaries.items():
        probabilities[classValue] = 0
        for i in range(len(classSummaries)):
            mean, stdev = classSummaries[i]
            x = inputVector[i]
            probabilities[classValue] += math.log10(calculateProbability(x, mean, stdev))
    return probabilities

#Determine the most likely class label
def predict(summaries, inputVector):
    probabilities = calculateClassProbabilities(summaries, inputVector)
    bestLabel, bestProb = None, -1
    for classValue, probability in probabilities.items():
        if bestLabel is None or probability > bestProb:
            bestProb = probability
            bestLabel = classValue
    return bestLabel

#Make predictions on the unseen data
def getPredictions(summaries, testSet):
    predictions = []
    for i in range(len(testSet)):
        result = predict(summaries, testSet[i])
        predictions.append(result)
    return predictions

#Compute performance in terms of F1 scores
def evaluate(testSet, predictions):
    tp1 = 0.0
    fp1 = 0.0
    fn1 = 0.0
    tp0 = 0.0
    fp0 = 0.0
    fn0 = 0.0
    for i in range(len(testSet)):
        if testSet[i][-1] == 0:
            if predictions[i] == 0:
                tp0 += 1.0
            else:
                fp1 += 1.0
                fn0 += 1.0
        else:
            if predictions[i] == 1:
                tp1 += 1.0
            else:
                fp0 += 1.0
                fn1 += 1.0
    p0 = tp0/(tp0+fp0)
    p1 = tp1/(tp1+fp1)
    r0 = tp0/(tp0+fn0)
    r1 = tp1/(tp1+fn1)
    print("F1 (Class 0): "+str((2.0*p0*r0)/(p0+r0)))
    print("F1 (Class 1): "+str((2.0*p1*r1)/(p1+r1)))

#Run the end-to-end baseline system
def baseline():
    random.seed(13)
    filename = 'readmission.csv'
    splitRatio = 0.80
    dataset = loadCsv(filename)
    trainingSet, valSet = splitDataset(dataset, splitRatio)
    print('Split '+str(len(dataset))+' rows into train = '+str(len(trainingSet))+' and test = '+str(len(valSet))+' rows.\n')
    # prepare model
    summaries = summarizeByClass(trainingSet)
    # test model
    predictions = getPredictions(summaries, valSet)
    evaluate(valSet, predictions)

#Train a new model on data
def training(data):
    #TODO: Implement 
    pass #TODO: Remove this line

#Load a trained model from file
def loading():
    pass #OPTIONAL: If you want to load a pre-trained model, implement this function and remove this line
  
#Use the model to produce predictions for a datset
def inference(model, data):
    #TODO: Implement
    pass #TODO: Remove this line


def pca(var_thres):
    """
    Performs PCA on the entire dataset except the last column.
    Input: Explained Variance ratio threshold that determines what proportion of the variance should be
    covered by the principal components.
    Output: New reduced dataset with readmission data restored as last column

    """
    data = loadCsv('readmission.csv')
    #is already an array
    #data = np.array(data)
    
    X = np.array(data[:,:-1])
    Y = np.array(data[:,-1])
    mean_vector = np.average(X,axis=0)

    new_X = X - mean_vector
    transposed = new_X.T
    cov_mat = np.cov(transposed)
    evalues, evectors = np.linalg.eig(cov_mat)

    sum_ = np.sum(evalues)
    explained_var = evalues/sum_
    pca_vector = np.matmul(new_X, evectors)
    
    cum_var = 0
    count = 0
    while cum_var < var_thres:
        cum_var += explained_var[count]
        reduced_data = pca_vector[:, :count]
        count += 1
    #print("PCA Threshold: " + str(var_thres))
    #print("Reduced to: " + str(count - 1) + " components")
    return concat_label(reduced_data, Y), count-1

#Helper method to add on readmission data column
def concat_label(array_1, array_2):
    new_array = np.insert(array_1, len(array_1[0]), array_2, axis=1)
    return new_array
    
result = pca(0.7)
#print(result[0][:, 13])
# print(len(result))
# print(len(result[0]))