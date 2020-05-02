# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 13:54:20 2019

@author: Jerold
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Nov  8 01:18:52 2019

@author: Jerold
"""

import pickle
import pandas as pd
import numpy as np
from Training import feature_evaluation 
from sklearn.metrics import accuracy_score
import sys

def test(file_name):
    
    with open("svm_model.pkl", 'rb') as file:
        svm_model = pickle.load(file)
    with open("knn_model.pkl", 'rb') as file:
        knn_model = pickle.load(file)
    with open("rforest_model.pkl", 'rb') as file:
        rforest_model = pickle.load(file)
    with open("logr_model.pkl", 'rb') as file:
        logr_model = pickle.load(file)
    with open("adaboost_model.pkl", 'rb') as file:
        adaboost_model = pickle.load(file)
    with open("mlp_model.pkl", 'rb') as file:
        mlp_model = pickle.load(file)
    with open("pca.pkl", 'rb') as file:
        pca = pickle.load(file)
    with open("scaling.pkl", 'rb') as file:
        pca_feat = pickle.load(file)
    
      
    # Reading csv file as cmd argument
    t_data = pd.read_csv(file_name, header=None)
    test_data = t_data.reindex(np.random.permutation(t_data.index))
    
    test_data.reset_index(drop=True, inplace=True)
    print(test_data)
    X_testing = test_data.iloc[:,:-1]
    Y_testing = test_data.iloc[:,-1]
#    
    # Calculating features for test data
    original_features = feature_evaluation(X_testing)
    
    # Getting the values from feature dataframe
    X_test = original_features.iloc[:].values
    
    # Scale test data using Created StandardScaler Model from training data
    X_trainn = pca_feat.transform(X_test) 
    
    # Transform Scaled data using Created PCA Model from training data
    X_tr = pca.transform(X_trainn)  
    
    svm_pred = svm_model.predict(X_tr)
    np.savetxt("SVM_prediction.csv", svm_pred, delimiter=",", fmt='%d')
    
    knn_pred = knn_model.predict(X_tr)
    np.savetxt("KNN_prediction.csv", knn_pred, delimiter=",", fmt='%d')
    
    rforest_pred = rforest_model.predict(X_tr)
    np.savetxt("RandomForest_prediction.csv", rforest_pred, delimiter=",", fmt='%d')
    
    logr_pred = logr_model.predict(X_tr)
    np.savetxt("LogisticRegression_prediction.csv", logr_pred, delimiter=",", fmt='%d')
    
    adaboost_pred = adaboost_model.predict(X_tr)
    np.savetxt("Adaboost_prediction.csv", adaboost_pred, delimiter=",", fmt='%d')
    
    mlp_pred = mlp_model.predict(X_tr)
    np.savetxt("MLP_prediction.csv", mlp_pred, delimiter=",", fmt='%d')
    
    print('SVM:', svm_model.score(X_tr, Y_testing ) )
    
    print('KNN:',knn_model.score(X_tr, Y_testing))
        
    print('R Forest:', rforest_model.score(X_tr, Y_testing))
    
    print('Log regression:',logr_model.score(X_tr, Y_testing))
        
    print('Adaboost:',adaboost_model.score(X_tr, Y_testing))
    
    print('MLP :',mlp_model.score(X_tr, Y_testing))

if __name__ == "__main__":
    
    file_name = sys.argv[1]
    test(file_name)
