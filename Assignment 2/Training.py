
import numpy as np
import pandas as pd
import os
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
import pickle
    
def preprocess():

    dataset1 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\mealData1.csv", usecols = [i for i in range(0,30)], header=None)
    dataset2 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\mealData2.csv", usecols = [i for i in range(0,30)], header=None)
    dataset3 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\mealData3.csv", usecols = [i for i in range(0,30)], header=None)
    dataset4 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\mealData4.csv", usecols = [i for i in range(0,30)], header=None)
    dataset5 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\mealData5.csv", usecols = [i for i in range(0,30)], header=None)
    
    dataset6 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\Nomeal1.csv", usecols = [i for i in range(0,30)], header=None)
    dataset7 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\Nomeal2.csv", usecols = [i for i in range(0,30)], header=None)
    dataset8 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\Nomeal3.csv", usecols = [i for i in range(0,30)], header=None)
    dataset9 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\Nomeal4.csv", usecols = [i for i in range(0,30)], header=None)
    dataset10 = pd.read_csv(r"D:\Fall 2019\DM\Assignment 2 - Group 34\MealNoMealData\Nomeal5.csv", usecols = [i for i in range(0,30)], header=None)
    
    dataset1_fill = dataset1.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset2_fill = dataset2.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset3_fill = dataset3.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset4_fill = dataset4.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset5_fill = dataset5.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    
    dataset6_fill = dataset6.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset7_fill = dataset7.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset8_fill = dataset8.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset9_fill = dataset9.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset10_fill = dataset10.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    
    dataset1_fill.reset_index(drop=True, inplace=True)
    dataset2_fill.reset_index(drop=True, inplace=True)
    dataset3_fill.reset_index(drop=True, inplace=True)
    dataset4_fill.reset_index(drop=True, inplace=True)
    dataset5_fill.reset_index(drop=True, inplace=True)
    
    dataset6_fill.reset_index(drop=True, inplace=True)
    dataset7_fill.reset_index(drop=True, inplace=True)
    dataset8_fill.reset_index(drop=True, inplace=True)
    dataset9_fill.reset_index(drop=True, inplace=True)
    dataset10_fill.reset_index(drop=True, inplace=True)
    
    
    dataset_meal = dataset1_fill.append([dataset2_fill, dataset3_fill,dataset4_fill, dataset5_fill])
    dataset_meal.reset_index(drop=True, inplace=True)
    dataset_meal['Class'] = 1
    
    dataset_nomeal = dataset6_fill.append([dataset7_fill, dataset8_fill,dataset9_fill, dataset10_fill])
    dataset_nomeal.reset_index(drop=True, inplace=True)
    dataset_nomeal['Class'] = 0

    original_data = dataset_meal.append([dataset_nomeal])
    original_data.reset_index(drop=True, inplace= True)
    
    dataset_meal = dataset_meal.drop(['Class'], axis=1)
    dataset_nomeal = dataset_nomeal.drop(['Class'], axis=1)
    
    feature_data_meal = feature_evaluation(dataset_meal)
    feature_data_nomeal = feature_evaluation(dataset_nomeal)
    feature_data_meal['Class'] = 1
    feature_data_nomeal['Class'] = 0
    
    feature_data = feature_data_meal.append([feature_data_nomeal])
    feature_data.reset_index(drop=True, inplace= True)
    
    return feature_data

def feature_evaluation(original_data):
    original_features = pd.DataFrame(index=np.arange(len(original_data)))
    
    # RMS
    original_features['Combined RMS'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined RMS'][i] = np.sqrt(np.mean(original_data.iloc[i,:])**2)
    
    # Distance Travelled    
    original_features['Combined Dist_travelled'] = np.nan
    for i in range(len(original_data)):
        row = original_data.loc[i,:].tolist()
        diff = []
        for j in range(1, len(row)):
            diff.append(abs(row[j] - row[j-1]))
        original_features['Combined Dist_travelled'][i] = np.round(np.sum(diff), 2)        
    
    # Auto correlation
    original_features['Combined autocorrelation'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined autocorrelation'][i] = ts.autocorrelation(np.array(original_data.iloc[i,:]), 1)
        
    # Velocity    
    original_features['Combined Velocity'] = np.nan
    for i in range(len(original_data)):
        row = original_data.loc[i,:].tolist()
        diff = []
        for j in range(1, len(row)):
            diff.append(abs(row[j] - row[j-1]))
        original_features['Combined Velocity'][i] = np.round(np.mean(diff), 2)        
    
    # Recurring values
    original_features['Combined Recurring values'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined Recurring values'][i] = ts.sum_of_reoccurring_values(original_data.loc[i,:])

    # Entropy
    original_features['Combined Entropy'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined Entropy'][i] = ts.sample_entropy(np.array(original_data.iloc[i,:]))
    
    # Skewness
    original_features['Combined Skewness'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined Skewness'][i] = ts.skewness(original_data.loc[i,:])
    
    # Min Max
    original_features['Combined Min'] = original_data.min(axis=1)
    original_features['Combined Max'] = original_data.max(axis=1)
    original_features.reset_index(drop=True, inplace=True)
    
    original_features['Combined MinMax'] = np.nan
    original_features['Combined MinMax'] = original_features['Combined Max'] - original_features['Combined Min']
    
    #CGM_Displacement
    original_features['Combined Displacement'] = np.nan
    for i in range(len(original_data)):
       c_list = original_data.loc[i,:].tolist()
       sum_=[]
       for j in range(1,len(c_list)):
           sum_.append(abs(c_list[j]-c_list[j-1]))
       original_features['Combined Displacement'][i] = np.round(np.sum(sum_),2)
    
    #CGM_Kurtosis
    original_features['Combined Kurtosis'] = np.nan
    for i in range(len(original_data)):
       original_features['Combined Kurtosis'][i] = ts.kurtosis(np.array(original_data.iloc[i,:]))
    
    #Recurr
    original_features['Combined Recur'] = np.nan
    for i in range(len(original_data)):
       original_features['Combined Recur'][i] = ts.ratio_value_number_to_time_series_length(np.array(original_data.iloc[i,:]))
    
    #Remove calculated columns
    del original_features['Combined Max']
    del original_features['Combined Min']
    
    original_features = original_features[['Combined Entropy','Combined Dist_travelled', 'Combined RMS', 'Combined Recurring values', 'Combined autocorrelation','Combined Velocity', 'Combined MinMax', 'Combined Skewness', 'Combined Displacement','Combined Kurtosis', 'Combined Recur']]
    
    return original_features

def Scaling(X_train):
    
    pca_feat = StandardScaler()
    X_train = pca_feat.fit_transform(X_train) 
    
    os.chdir(r"D:\Fall 2019\DM\Assignment 2 - Group 34")

    pkl_filename = "scaling.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pca_feat, file)

    return X_train

def Pca(X_train):
    
    pca = PCA(0.95)
    pca_weights = pca.fit(X_train)
    dot = pca.fit_transform(X_train)
    
    new_file = pd.DataFrame(data=(pca.components_).T)
    new_file.to_csv('EigenVectors.csv',index = False, header = False)
    
    pkl_filename = "pca.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(pca_weights, file)
 
    return dot

def svm_classifier(X_train, Y_train):
    
    clf_svm=svm.SVC(kernel = 'rbf', gamma=0.009, C=1)
    clf_svm.fit(X_train,Y_train)
    
    pkl_filename = "svm_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_svm, file)


def knn_classifier(X_train, Y_train):
    
    clf_knn = KNeighborsClassifier(n_neighbors=10, p=2)
    clf_knn.fit(X_train, Y_train)
    
    pkl_filename = "knn_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_knn, file)


def rforest_classifier(X_train, Y_train):
    
    clf_rforest=RandomForestClassifier(n_estimators=100, max_depth=4, max_features=3)
    clf_rforest.fit(X_train,Y_train)

    pkl_filename = "rforest_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_rforest, file)

def logregression_classifier(X_train, Y_train):
    
    clf_logr = LogisticRegression(penalty = 'l2' , C=0.01)
    clf_logr.fit(X_train, Y_train)
    
    pkl_filename = "logr_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_logr, file)


def Adaboost_classifier(X_train, Y_train):
    
    clf_adb = AdaBoostClassifier(n_estimators=10, random_state=0)
    clf_adb.fit(X_train, Y_train)
    
    pkl_filename = "adaboost_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(clf_adb, file)


def MLP_classifier(X_train, Y_train):
    
    mlp_clf = MLPClassifier(alpha=1, max_iter=1000)
    mlp_clf.fit(X_train, Y_train)
    
    pkl_filename = "mlp_model.pkl"
    with open(pkl_filename, 'wb') as file:
        pickle.dump(mlp_clf, file)

    
def Classification(original_features):
    
    X=original_features.iloc[:,:-1].values
    Y=original_features.iloc[:,-1].values
    
    X_train = Scaling(X)        
    X_train = Pca(X_train)
    
    svm_classifier(X_train, Y)
    knn_classifier(X_train, Y)
    rforest_classifier(X_train, Y)
    logregression_classifier(X_train, Y)
    Adaboost_classifier(X_train, Y)
    MLP_classifier(X_train, Y)

if __name__ == "__main__":
    
    original_data = preprocess()
    Classification(original_data)
     
