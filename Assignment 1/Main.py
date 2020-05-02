# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:17:02 2019

@author: Jerold
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
import tsfresh.feature_extraction.feature_calculators as ts

dataset1 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\CGMSeriesLunchPat1_new.csv")
dataset2 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\CGMSeriesLunchPat2_new.csv")
dataset3 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\CGMSeriesLunchPat3_new.csv")
dataset4 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\CGMSeriesLunchPat4_new.csv")
dataset5 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\CGMSeriesLunchPat5_new.csv")


dataset = dataset1.append([dataset2, dataset3,dataset4,dataset5])
dataset.reset_index(drop=True, inplace=True)


###################### Time Series Features ######################

time_features=pd.DataFrame()

# Mean
time_features['CGM_total_mean'] = dataset.mean(axis=1)
time_features['CGM_mean_1'] = dataset.iloc[:,0:6].mean(axis = 1)
time_features['CGM_mean_2'] = dataset.iloc[:,6:12].mean(axis = 1)
time_features['CGM_mean_3'] = dataset.iloc[:,12:18].mean(axis = 1)
time_features['CGM_mean_4'] = dataset.iloc[:,18:24].mean(axis = 1)
time_features['CGM_mean_5'] = dataset.iloc[:,24:].mean(axis = 1)

# Variance
time_features['CGM_total_variance'] = dataset.var(axis=1)
time_features['CGM_variance_1'] = dataset.iloc[:,0:6].var(axis = 1)
time_features['CGM_variance_2'] = dataset.iloc[:,6:12].var(axis = 1)
time_features['CGM_variance_3'] = dataset.iloc[:,12:18].var(axis = 1)
time_features['CGM_variance_4'] = dataset.iloc[:,18:24].var(axis = 1)
time_features['CGM_variance_5'] = dataset.iloc[:,24:].var(axis = 1)

# Standard Deviation
time_features['CGM_total_SD'] = dataset.std(axis=1)
time_features['CGM_SD_1'] = dataset.iloc[:,0:6].std(axis = 1)
time_features['CGM_SD_2'] = dataset.iloc[:,6:12].std(axis = 1)
time_features['CGM_SD_3'] = dataset.iloc[:,12:18].std(axis = 1)
time_features['CGM_SD_4'] = dataset.iloc[:,18:24].std(axis = 1)
time_features['CGM_SD_5'] = dataset.iloc[:,24:].std(axis = 1)

# Min Max

time_features['CGM_Min'] = dataset.min(axis=1)
time_features['CGM_Max'] = dataset.max(axis=1)
time_features.reset_index(drop=True, inplace=True)

time_features['CGM_MinMax'] = np.nan
time_features['CGM_MinMax'] = time_features['CGM_Max'] - time_features['CGM_Min']

# Entropy
time_features['CGM_Entropy'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_Entropy'][i] = ts.sample_entropy(np.array(dataset.iloc[i,:]))

# Skewness
time_features['CGM_Skewness'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_Skewness'][i] = ts.skewness(dataset.loc[i,:])

# First Max
time_features['CGM_First_Max'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_First_Max'][i] = ts.first_location_of_maximum(np.array(dataset.iloc[i,:]))

# First Min
time_features['CGM_First_Min'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_First_Min'][i] = ts.first_location_of_minimum(np.array(dataset.iloc[i,:]))

# Last Max
time_features['CGM_Last_Max'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_Last_Max'][i] = ts.last_location_of_maximum(np.array(dataset.iloc[i,:]))
    
# Last Min
time_features['CGM_Last_Min'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_Last_Min'][i] = ts.last_location_of_minimum(np.array(dataset.iloc[i,:]))

# Number of peaks
time_features['CGM_number_of_peaks'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_number_of_peaks'][i] = ts.number_peaks(np.array(dataset.iloc[i,:]), 2)

# Velocity    
time_features['CGM_Velocity'] = np.nan
for i in range(len(dataset)):
    row = dataset.loc[i,:].tolist()
    diff = []
    for j in range(1, len(row)):
        diff.append(abs(row[j] - row[j-1]))
    time_features['CGM_Velocity'][i] = np.round(np.mean(diff), 2)        
    
# Distance Travelled
time_features['CGM_Dist_travelled'] = np.nan
for i in range(len(dataset)):
    row = dataset.loc[i,:].tolist()
    diff = []
    for j in range(1, len(row)):
        diff.append(abs(row[j] - row[j-1]))
    time_features['CGM_Dist_travelled'][i] = np.round(np.sum(diff), 2)        

# Auto correlation
time_features['CGM_autocorrelation'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_autocorrelation'][i] = ts.autocorrelation(np.array(dataset.iloc[i,:]), 1)

# RMS    
time_features['CGM_RMS'] = np.nan
for i in range(len(dataset)):
    time_features['CGM_RMS'][i] = np.sqrt(np.mean(dataset.iloc[i,:])**2)
    
# AUC
#from sklearn import metrics
#time_features['CGM_AUC'] = np.nan
#for i in range(len(dataset)):
#    arr = []
#    arr += 31 * [time_features['CGM_Min'][i]]
#    fpr, tpr, thresholds= metrics.roc_curve(np.array(dataset.iloc[i,:]), arr,pos_label=1 )
#    time_features['CGM_AUC'][i] = metrics.auc(fpr, tpr)

    
###################### PCA ######################

    
time_features = time_features[['CGM_total_mean','CGM_total_variance','CGM_total_SD','CGM_MinMax', 'CGM_Entropy', 'CGM_Skewness', 'CGM_First_Max', 'CGM_First_Min', 'CGM_Last_Max','CGM_Last_Min'
                             ,'CGM_number_of_peaks','CGM_Velocity','CGM_Dist_travelled','CGM_autocorrelation', 'CGM_RMS']]

from sklearn.preprocessing import StandardScaler
PCA_features = StandardScaler().fit_transform(time_features)


from sklearn.decomposition import PCA 
pca = PCA(n_components = 5)
principalcomponents = pca.fit_transform(PCA_features)
principalcomponentsdf = pd.DataFrame(data = principalcomponents, columns = ['PC 1', 'PC 2','PC 3','PC 4','PC 5'])


print(pca.explained_variance_)
print(pca.components_)

# maximum feature attribute in each row 
for i in pca.components_:
    print(max(i))

# maximum feature attribute in first row
sorted(pca.components_[0])

# location of maximum feature attribute in each row
for i,row in enumerate(pca.components_):
    print(sorted(range(len(pca.components_[i])), key=lambda j: pca.components_[i][j])[-1:])

# location of top 5 maximum values in first row
sorted(range(len(pca.components_[0])), key=lambda i: pca.components_[0][i])[-5:]

n_samples = time_features.shape[0]
cov_matrix = np.dot(time_features.T, time_features) / n_samples
eigenvalues = pca.explained_variance_
for eigenvalue, eigenvector in zip(eigenvalues, pca.components_):    
    print(np.dot(eigenvector.T, np.dot(cov_matrix, eigenvector)))
    print(eigenvalue)
# Writinf the feature matrix to a csv file
time_features.to_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed\time_features.csv", index=False)
