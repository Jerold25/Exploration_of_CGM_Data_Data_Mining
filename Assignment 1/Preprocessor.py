# -*- coding: utf-8 -*-
"""
Created on Sat Sep 28 16:50:08 2019

@author: Jerold
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
#os.chdir(r"C:\Users\motth\Desktop\Udemy\Machine_Learning\Machine Learning A-Z Template Folder\Part 2 - Regression")

dataset = pd.read_csv(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\CGMSeriesLunchPat1.csv")
X = dataset.iloc[:,:].values

from sklearn.preprocessing import Imputer
pre_process = Imputer(missing_values='NaN', strategy = 'mean', axis=1)
pre_process = pre_process.fit(X[:,:])
X[:,:] = pre_process.transform(X[:,:])

column=[]
for i in dataset.columns:
    column.append(i)

new_file = pd.DataFrame(data=X[0:,0:])
new_file.columns = column

new_file.iloc[:,:] = np.round(new_file.iloc[:,:],2)

os.chdir(r"D:\Fall 2019\DM\Assignment 1\DataFolder\DataFolder\Pre Processed")
new_file.to_csv('CGMSeriesLunchPat4'+'_new.csv',index = False)

os.getcwd()