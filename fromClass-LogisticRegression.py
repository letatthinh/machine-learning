'''
March 1, 2023, updated February 12, 2025
Logistic Regression, adapted from for Ch8 of Machine Learning in Action
@author: Clif Baldwin
'''

import numpy as np
import pandas as pd

penguins = pd.read_csv("https://raw.githubusercontent.com/allisonhorst/palmerpenguins/master/inst/extdata/penguins.csv",delimiter=",")

# Drop rows with NAs
lmPenguins = penguins.dropna(axis=0)

labels = lmPenguins.sex.replace(to_replace=['male', 'female'], value=[0, 1]).to_numpy()

independentVars = lmPenguins[['bill_length_mm','bill_depth_mm','flipper_length_mm','body_mass_g']].to_numpy()

def autoNorm(dataSet):
    minVals = dataSet.min(0) 
    maxVals = dataSet.max(0) 
    ranges = maxVals - minVals 
    normDataSet = np.zeros(np.shape(dataSet))
    m = dataSet.shape[0] 
    normDataSet = dataSet - np.tile(minVals, (m, 1)) 
    normDataSet = normDataSet/np.tile(ranges, (m, 1)) 
    return normDataSet, ranges, minVals
    

def sigmoid(inX):
    return 1.0 / (1 + np.exp(-inX))
    
# Gradient Descent algorithm
def gradAscent(dataMatIn, classLabels):
    dataMatrix = np.mat(dataMatIn)             
    labelMat = np.mat(classLabels).transpose() 
    m, n = np.shape(dataMatrix)
    alpha = 0.001
    maxCycles = 500
    weights = np.ones((n, 1))
    for k in range(maxCycles):              
        h = sigmoid(dataMatrix*weights)     
        error = (labelMat - h)              
        weights = weights + alpha * dataMatrix.transpose()* error 
    return weights

def classifyVector(inX, weights):
    prob = sigmoid(sum(inX*weights))
    # Note: I set 0.5 as the divider between male (0) and female (0). 
    if prob > 0.5: return 1.0 # You can change it if you think 0.5 is not the best
    else: return 0.0
    
def predictVector(X_test, weights):
    prob = sigmoid(X_test*beta)
    return np.where(prob > 0.5, 1, 0).flatten()
    

normVars, ranges, minVals = autoNorm(independentVars)

# Need to add a column of 1's for beta_0 (i.e. the bias)
normVars = np.insert(normVars, 0, 1, axis=1)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(normVars, labels, test_size = 0.2, random_state = 0)


# If train_test_split does not work for you,
#  You can split the data the hard way
import random
def train_test_split(data, test_size=0.2):
    """
    Splits data into training and testing sets.

    Args:
        data: A list or tuple containing the dataset.
        test_size: The proportion of the data to include in the test set (0.0 to 1.0).

    Returns:
        A tuple containing two lists: (training_data, testing_data).
    """
    if not (0.0 < test_size < 1.0):
        raise ValueError("test_size must be between 0.0 and 1.0")

    data_copy = list(data)
    random.shuffle(data_copy)

    split_index = int(len(data_copy) * (1 - test_size))
    training_data = data_copy[:split_index]
    testing_data = data_copy[split_index:]

    return training_data, testing_data

# Example usage:
# train_data, test_data = train_test_split(normVars, test_size=0.3)






beta = gradAscent(X_train, y_train)

    
y_hat = predictVector(X_test, beta)

# See https://www.w3schools.com/python/python_ml_confusion_matrix.asp
from sklearn import metrics 
import matplotlib.pyplot as plt 
confusion_matrix = metrics.confusion_matrix(y_test, y_hat) 
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = ['Male', 'Female']) 
cm_display.plot()
plt.show() 

true_negatives = metrics.confusion_matrix(y_test, y_hat)[0,0] # True males
true_positives = metrics.confusion_matrix(y_test, y_hat)[1,1] # True females
false_positives = metrics.confusion_matrix(y_test, y_hat)[0,1] # predicted female but actually male
false_negatives = metrics.confusion_matrix(y_test, y_hat)[1,0] # predicted male but actually female

sensitivity = true_positives / (true_positives + false_negatives)
specificity = true_negatives / (true_negatives + false_positives)
accuracy = (true_positives + true_negatives) / sum(sum(metrics.confusion_matrix(y_test, y_hat)))

print("Accuracy is " + str(accuracy))
print("Sensitivity is " + str(sensitivity) + ", in this case, percentage of correct females")
print("Specificity is " + str(specificity)  + ", in this case, percentage of correct males")
