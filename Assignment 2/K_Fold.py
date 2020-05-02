
import numpy as np
import pandas as pd
import os
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import recall_score
from sklearn.metrics import precision_score
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier


test_labels_svm = []
accuracy_svm = []
pred_labels_svm = []
f1score_svm = []
recall_svm = []
precision_svm = []

test_labels_knn = []
accuracy_knn = []
pred_labels_knn = []
f1score_knn = []
recall_knn = []
precision_knn = []

test_labels_rforest = []
accuracy_rforest = []
pred_labels_rforest = []
f1score_rforest = []
recall_rforest = []
precision_rforest= []

test_labels_logr = []
accuracy_logr = []
pred_labels_logr = []
f1score_logr = []
recall_logr = []
precision_logr = []

test_labels_adb = []
accuracy_adb = []
pred_labels_adb = []
f1score_adb = []
recall_adb = []
precision_adb = []

test_labels_mlp = []
accuracy_mlp = []
pred_labels_mlp = []
f1score_mlp = []
recall_mlp = []
precision_mlp = []
    
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
    
    os.chdir(r"D:\Fall 2019\DM\Assignment 2 - Group 34")
    original_data.to_csv('OriginalData.csv',index = False, header = False)
    
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
    
    # Recurring values
    original_features['Combined Recurring values'] = np.nan
    for i in range(len(original_data)):
        original_features['Combined Recurring values'][i] = ts.sum_of_reoccurring_values(original_data.loc[i,:])


    #Recurr
    original_features['Combined Recur'] = np.nan
    for i in range(len(original_data)):
       original_features['Combined Recur'][i] = ts.ratio_value_number_to_time_series_length(np.array(original_data.iloc[i,:]))
    
    #Remove calculated columns
    del original_features['Combined Max']
    del original_features['Combined Min']
    
    original_features = original_features[['Combined Entropy','Combined Dist_travelled', 'Combined RMS', 'Combined autocorrelation','Combined Velocity', 'Combined Recurring values', 'Combined MinMax', 'Combined Skewness', 'Combined Displacement','Combined Kurtosis', 'Combined Recur']]
    
    return original_features

def Scaling(X_train, X_test):
    
    pca_feat = StandardScaler()
    X_train = pca_feat.fit_transform(X_train) 
    X_test = pca_feat.transform(X_test)

    return X_train, X_test

def Pca(X_train, X_test):
    
    pca = PCA(0.95)
    dot = pca.fit_transform(X_train)
    dot2 = pca.transform(X_test)  
    
    return dot, dot2

def svm_classifier(X_train, Y_train, X_test, Y_test):
    
    clf_svm=svm.SVC(kernel = 'rbf', gamma=0.009, C=1)
    clf_svm.fit(X_train,Y_train)
    Y_pred_svm=clf_svm.predict(X_test)
    test_labels_svm.extend(Y_test)
    pred_labels_svm.extend(Y_pred_svm)
    accuracy_svm.append(accuracy_score(test_labels_svm,pred_labels_svm))
    f1score_svm.append(f1_score(test_labels_svm,pred_labels_svm))
    recall_svm.append(recall_score(test_labels_svm,pred_labels_svm))
    precision_svm.append(precision_score(test_labels_svm,pred_labels_svm))
    
    return accuracy_svm, f1score_svm, recall_svm, precision_svm

def knn_classifier(X_train, Y_train, X_test, Y_test):
    
    clf_knn = KNeighborsClassifier(n_neighbors=10, p=2)
    clf_knn.fit(X_train, Y_train)
    Y_pred_knn = clf_knn.predict(X_test)
    test_labels_knn.extend(Y_test)
    pred_labels_knn.extend(Y_pred_knn)
    accuracy_knn.append(accuracy_score(test_labels_knn,pred_labels_knn))
    f1score_knn.append(f1_score(test_labels_knn,pred_labels_knn))
    recall_knn.append(recall_score(test_labels_knn,pred_labels_knn))
    precision_knn.append(precision_score(test_labels_knn,pred_labels_knn))
    
    return accuracy_knn, f1score_knn, recall_knn, precision_knn

def rforest_classifier(X_train, Y_train, X_test, Y_test):
    
    clf_rforest=RandomForestClassifier(n_estimators=100, max_depth=4, max_features=3)
    clf_rforest.fit(X_train,Y_train)
    Y_pred_rforest=clf_rforest.predict(X_test)
    test_labels_rforest.extend(Y_test)
    pred_labels_rforest.extend(Y_pred_rforest)
    accuracy_rforest.append(accuracy_score(test_labels_rforest,pred_labels_rforest))
    f1score_rforest.append(f1_score(test_labels_rforest,pred_labels_rforest))
    recall_rforest.append(recall_score(test_labels_rforest,pred_labels_rforest))
    precision_rforest.append(precision_score(test_labels_rforest,pred_labels_rforest))
    
    return accuracy_rforest, f1score_rforest, recall_rforest, precision_rforest

def logregression_classifier(X_train, Y_train, X_test, Y_test):
    
    clf_logr = LogisticRegression(penalty = 'l2' , C=0.01)
    clf_logr.fit(X_train, Y_train)
    Y_pred_logr = clf_logr.predict(X_test)
    test_labels_logr.extend(Y_test)
    pred_labels_logr.extend(Y_pred_logr)
    accuracy_logr.append(accuracy_score(test_labels_logr,pred_labels_logr))
    f1score_logr.append(f1_score(test_labels_logr,pred_labels_logr))
    recall_logr.append(recall_score(test_labels_logr,pred_labels_logr))
    precision_logr.append(precision_score(test_labels_logr,pred_labels_logr))
    
    return accuracy_logr, f1score_logr, recall_logr, precision_logr


def Adaboost_classifier(X_train, Y_train, X_test, Y_test):
    
    clf_adb = AdaBoostClassifier(n_estimators=10, random_state=0)
    clf_adb.fit(X_train, Y_train)
    Y_pred_adb = clf_adb.predict(X_test)
    test_labels_adb.extend(Y_test)
    pred_labels_adb.extend(Y_pred_adb)
    accuracy_adb.append(accuracy_score(test_labels_adb,pred_labels_adb))
    f1score_adb.append(f1_score(test_labels_adb,pred_labels_adb))
    recall_adb.append(recall_score(test_labels_adb,pred_labels_adb))
    precision_adb.append(precision_score(test_labels_adb,pred_labels_adb))

    return accuracy_adb, f1score_adb, recall_adb, precision_adb


def MLP_classifier(X_train, Y_train, X_test, Y_test):
    
    mlp_clf = MLPClassifier(alpha=1, max_iter=1000)
    mlp_clf.fit(X_train, Y_train)
    Y_pred_mlp = mlp_clf.predict(X_test)
    test_labels_mlp.extend(Y_test)
    pred_labels_mlp.extend(Y_pred_mlp)
    accuracy_mlp.append(accuracy_score(test_labels_mlp,pred_labels_mlp))
    f1score_mlp.append(f1_score(test_labels_mlp,pred_labels_mlp))
    recall_mlp.append(recall_score(test_labels_mlp,pred_labels_mlp))
    precision_mlp.append(precision_score(test_labels_mlp,pred_labels_mlp))
    

    return accuracy_mlp, f1score_mlp, recall_mlp, precision_mlp

     
def Classification(original_features):
    
    X=original_features.iloc[:,:-1].values
    Y=original_features.iloc[:,-1].values
        
    # K-Fold
    skf = KFold(n_splits=6, random_state=42, shuffle=True)
    
    for train_index,test_index in skf.split(X, Y):
        
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        
        X_train, X_test = Scaling(X_train, X_test)        
        X_train, X_test = Pca(X_train, X_test)
        
        accuracy_svm, f1score_svm, recall_svm, precision_svm = svm_classifier(X_train, Y_train, X_test, Y_test)
        accuracy_knn, f1score_knn, recall_knn, precision_knn = knn_classifier(X_train, Y_train, X_test, Y_test)
        accuracy_rforest, f1score_rforest, recall_rforest, precision_rforest  = rforest_classifier(X_train, Y_train, X_test, Y_test)
        accuracy_logr, f1score_logr, recall_logr, precision_logr  = logregression_classifier(X_train, Y_train, X_test, Y_test)
        accuracy_adb, f1score_adb, recall_adb, precision_adb  = Adaboost_classifier(X_train, Y_train, X_test, Y_test)
        accuracy_mlp, f1score_mlp, recall_mlp, precision_mlp  = MLP_classifier(X_train, Y_train, X_test, Y_test)

    print('SVM:\n','Accuracy:',np.mean(accuracy_svm),'\n F1-Score:', np.mean(f1score_svm),'\n Recall:  ', np.mean(recall_svm),'\n Precision:', np.mean(precision_svm))
    print('KNN:\n','Accuracy:',np.mean(accuracy_knn),'\n F1-Score:', np.mean(f1score_knn),'\n Recall:  ', np.mean(recall_knn),'\n Precision:', np.mean(precision_knn))
    print('R-Forest:\n','Accuracy:',np.mean(accuracy_rforest),'\n F1-Score:', np.mean(f1score_rforest),'\n Recall:  ', np.mean(recall_rforest),'\n Precision:', np.mean(precision_rforest))
    print('Log-Regression:\n','Accuracy:',np.mean(accuracy_logr),'\n F1-Score:', np.mean(f1score_logr),'\n Recall:  ', np.mean(recall_logr),'\n Precision:', np.mean(precision_logr))
    print('Ada Boost:\n','Accuracy:',np.mean(accuracy_adb),'\n F1-Score:', np.mean(f1score_adb),'\n Recall:  ', np.mean(recall_adb),'\n Precision:', np.mean(precision_adb))
    print('MLP:\n','Accuracy:',np.mean(accuracy_mlp),'\n F1-Score:', np.mean(f1score_mlp),'\n Recall:  ', np.mean(recall_mlp),'\n Precision:', np.mean(precision_mlp))
    

if __name__ == "__main__":
    
    original_data = preprocess()
    Classification(original_data)
     
