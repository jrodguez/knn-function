#Imports

import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

#Euclidean distance calculator
# This function will simply return the euclidean distance between a row in the intput data to be classified.

def euclidean_distance(train_point, test_point):
    """This function calculates the euclidean distance between two points"""
    return np.sqrt(np.sum((train_point - test_point)**2))

#Sorted list of euclidean distances
def euclidean_list(X_train, y_train, test_point, k):
    """This function returns a sorted list of euclidean distances from a test point to all training points"""
    
    temp_list = [] # temporary list to store euclidean distances and other corresponding data
    
    for train_point in range(X_train.shape[0]): # calls on every training point for use in the euclidean_distance
        
        ED = euclidean_distance(X_train[train_point], test_point) #calculates euclidean distance
        temp_list.append([X_train[train_point], ED, y_train[train_point]]) # appends data to the temporary list
        
        EL = pd.DataFrame(temp_list, columns = ['training point', 'distance', 'class']) #converts temporary list to a dataframe
        sort_EL = EL.sort_values('distance')[:k] # sorts the dataframe based on k nearest neighbors
        
    return sort_EL

#knn wrapping function
def KNN(X_train, y_train, X_test, k):
    """This function returns the class prediction from the list of sorted Euclidean distances based on the KNN"""
    test_point_prediction = [] # empty list to append the predicted classifications
    
    for test_point in range(X_test.shape[0]): # calling on every test point for use in euclidean_list
        ED_list = euclidean_list(X_train, y_train, X_test[test_point], k)
        
        # this is the portion that actually chooses the class based on the greatest number of nearest neighbors
        test_point_label = stats.mode(ED_list['class'])[0]
        test_point_prediction.append([X_test[test_point], test_point_label]) 
        
        # Here, the test points and their corresponding predicted classifications are put into a data frame
        Classification = pd.DataFrame(test_point_prediction, columns = ['test point', 'classification'])
    
    return Classification 

#Selecting proper k value uses next two functions

#Accuracy
def accuracy(X_train, y_train, X_test, y_test, k):
    knn = KNN(X_train, y_train, X_test, k)
    acc = 100 * float((y_test == knn['classification']).sum()) / knn.shape[0]
    return acc

def k_select(X_train, y_train, X_test, y_test, k_range):
    
    accuracy_list = []
    
    for k in k_range:
        accuracy_list.append(accuracy(X_train, y_train, X_test, y_test, k))
    
    k_plot = plt.plot(k_range, accuracy_list)
    plt.xlabel('k value')
    plt.ylabel('% accuracy')

    return k_plot
