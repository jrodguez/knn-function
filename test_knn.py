import knn
import pandas as pd
import numpy as np
from collections import Counter
from scipy import stats
import matplotlib.pyplot as plt

#Euclidean Distance Calculator

def test_euclidean_distance():
    """This is a test function for euclidean_distance """
    #The euclidean distance between the same point should be zero
    assert int(knn.euclidean_distance(np.array([1,1]), np.array([1,1]))) == 0, "Calculation incorrect"
               
    return 

#Sorted List of euclidean distances

def test_euclidean_list():
    """This is a test function for euclidean_list """
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_train = pd.read_csv('atomradii.csv')
    X_train = df_train.iloc[:, :-2].values
    y_train = df_train.iloc[:, 3].values
    
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_test = pd.read_csv('testing.csv')
    X_test = df_test.iloc[:, :-2].values
    y_test = df_test.iloc[:, 3].values
    
    test_point = np.array([1,1])
    k = 5
    
    temp_list = [] # temporary list to store euclidean distances and other corresponding data
    
    for train_point in range(X_train.shape[0]): # calls on every training point for use in the euclidean_distance
        
        ED = knn.euclidean_distance(X_train[train_point], test_point) #calculates euclidean distance
        temp_list.append([X_train[train_point], ED, y_train[train_point]]) # appends data to the temporary list
        
        EL = pd.DataFrame(temp_list, columns = ['training point', 'distance', 'class']) #converts temporary list to a dataframe
        sort_EL = EL.sort_values('distance')[:k] # sorts the dataframe based on k nearest neighbors
        
        #Asserting that there should be no null values in dataframe if sorted correctly
        assert sort_EL.isnull().values.any() == False,  "null (NaN) values are present, not sorting correctly"
        
    return 


#KNN Function

def test_KNN():
    """This is a test function for KNN """
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_train = pd.read_csv('atomradii.csv')
    X_train = df_train.iloc[:, :-2].values
    y_train = df_train.iloc[:, 3].values
    
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_test = pd.read_csv('testing.csv')
    X_test = df_test.iloc[:, :-2].values
    y_test = df_test.iloc[:, 3].values
    
    test_point = np.array([1,1])
    k = 5
    
    test_point_prediction = [] # empty list to append the predicted classifications
    
    for test_point in range(X_test.shape[0]): # calling on every test point for use in euclidean_list
        ED_list = knn.euclidean_list(X_train, y_train, X_test[test_point], k)
        
        # this is the portion that actually chooses the class based on the greatest number of nearest neighbors
        test_point_label = stats.mode(ED_list['class'])[0]
        test_point_prediction.append([X_test[test_point], test_point_label]) 
        
        # Here, the test points and their corresponding predicted classifications are put into a data frame
        Classification = pd.DataFrame(test_point_prediction, columns = ['test point', 'classification'])
        
        #Asserting that there should be no null values in dataframe if classified correctly
        assert Classification.isnull().values.any() == False,  "null (NaN) values are present, not classifying correctly"
        
        return

#Accuracy

def test_accuracy():
    """This is a test function for accuracy """
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_train = pd.read_csv('atomradii.csv')
    X_train = df_train.iloc[:, :-2].values
    y_train = df_train.iloc[:, 3].values
    
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_test = pd.read_csv('testing.csv')
    X_test = df_test.iloc[:, :-2].values
    y_test = df_test.iloc[:, 3].values
    
    test_point = np.array([1,1])
    k = 5
    
    neighbor = knn.KNN(X_train, y_train, X_test, k)
    acc = 100 * float((y_test == neighbor['classification']).sum()) / neighbor.shape[0]
    
    assert isinstance(acc, float), "returned accuracy is not a float value" 
    
    return 


#K selector

def test_k_select():
    """This is a test function for k_select """
     # Setting up our training data and seperating based on the x,y coordinate and type
    df_train = pd.read_csv('atomradii.csv')
    X_train = df_train.iloc[:, :-2].values
    y_train = df_train.iloc[:, 3].values
    
    # Setting up our training data and seperating based on the x,y coordinate and type
    df_test = pd.read_csv('testing.csv')
    X_test = df_test.iloc[:, :-2].values
    y_test = df_test.iloc[:, 3].values
    
    test_point = np.array([1,1])
    k_range = list(range(1,25))
    
    accuracy_list = []
    
    for k in k_range:
        accuracy_list.append(knn.accuracy(X_train, y_train, X_test, y_test, k))
        
    # Asserting that the returned list is not empty   
    assert len(accuracy_list) >0, "List is empty"
    
    
    return 
