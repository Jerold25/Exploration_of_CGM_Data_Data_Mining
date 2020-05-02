
import pandas as pd
import numpy as np
import tsfresh.feature_extraction.feature_calculators as ts
from sklearn.preprocessing import StandardScaler

from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import silhouette_score
from sklearn.cluster import DBSCAN



def preprocess():

    dataset1 = pd.read_csv(r"MealNoMealData\mealData1.csv", usecols = [i for i in range(0,30)], header=None)
    dataset2 = pd.read_csv(r"MealNoMealData\mealData2.csv", usecols = [i for i in range(0,30)], header=None)
    dataset3 = pd.read_csv(r"MealNoMealData\mealData3.csv", usecols = [i for i in range(0,30)], header=None)
    dataset4 = pd.read_csv(r"MealNoMealData\mealData4.csv", usecols = [i for i in range(0,30)], header=None)
    dataset5 = pd.read_csv(r"MealNoMealData\mealData5.csv", usecols = [i for i in range(0,30)], header=None)
    
    dataset_meal = dataset1.append([dataset2, dataset3, dataset4, dataset5])
    dataset_meal.reset_index(drop=True, inplace=True)
    
    null_Index = pd.isnull(dataset_meal).all(1).nonzero()[0].tolist()

    dataset_meal = dataset_meal.dropna(axis=0, how='all').interpolate(method ='linear', limit_direction ='both')
    dataset_meal.reset_index(drop=True, inplace=True)
    
    dataset6 = pd.read_csv(r"MealNoMealData\mealAmountData1.csv", nrows = len(dataset1), header=None)
    dataset7 = pd.read_csv(r"MealNoMealData\mealAmountData2.csv", nrows = len(dataset2), header=None)
    dataset8 = pd.read_csv(r"MealNoMealData\mealAmountData3.csv", nrows = len(dataset3), header=None)
    dataset9 = pd.read_csv(r"MealNoMealData\mealAmountData4.csv", nrows = len(dataset4), header=None)
    dataset10 = pd.read_csv(r"MealNoMealData\mealAmountData5.csv", nrows = len(dataset5), header=None)
    
    dataset_carbs = dataset6.append([dataset7, dataset8, dataset9, dataset10])
    dataset_carbs.reset_index(drop=True, inplace=True)
    
    
    for i in null_Index:
        dataset_carbs = dataset_carbs.drop(i)
        
    dataset_carbs.reset_index(drop=True, inplace=True)
    
    return dataset_meal, dataset_carbs

def feature_evaluation(dataset_meal):
    
    original_features = pd.DataFrame(index=np.arange(len(dataset_meal)))
    
    # RMS
    original_features['Combined RMS'] = np.nan
    for i in range(len(dataset_meal)):
        original_features['Combined RMS'][i] = np.sqrt(np.mean(dataset_meal.iloc[i,:])**2)
    
    # Distance Travelled    
    original_features['Combined Dist_travelled'] = np.nan
    for i in range(len(dataset_meal)):
        row = dataset_meal.loc[i,:].tolist()
        diff = []
        for j in range(1, len(row)):
            diff.append(abs(row[j] - row[j-1]))
        original_features['Combined Dist_travelled'][i] = np.round(np.sum(diff), 2)        
    
    # Auto correlation
    original_features['Combined autocorrelation'] = np.nan
    for i in range(len(dataset_meal)):
        original_features['Combined autocorrelation'][i] = ts.autocorrelation(np.array(dataset_meal.iloc[i,:]), 1)
        
    # Velocity    
    original_features['Combined Velocity'] = np.nan
    for i in range(len(dataset_meal)):
        row = dataset_meal.loc[i,:].tolist()
        diff = []
        for j in range(1, len(row)):
            diff.append(abs(row[j] - row[j-1]))
        original_features['Combined Velocity'][i] = np.round(np.mean(diff), 2)        
    
    # Entropy
    original_features['Combined Entropy'] = np.nan
    for i in range(len(dataset_meal)):
        original_features['Combined Entropy'][i] = ts.sample_entropy(np.array(dataset_meal.iloc[i,:]))
    
    # Skewness
    original_features['Combined Skewness'] = np.nan
    for i in range(len(dataset_meal)):
        original_features['Combined Skewness'][i] = ts.skewness(dataset_meal.loc[i,:])
    
    # Min Max
    original_features['Combined Min'] = dataset_meal.min(axis=1)
    original_features['Combined Max'] = dataset_meal.max(axis=1)
    original_features.reset_index(drop=True, inplace=True)
    
    original_features['Combined MinMax'] = np.nan
    original_features['Combined MinMax'] = original_features['Combined Max'] - original_features['Combined Min']
    
    #CGM_Displacement
    original_features['Combined Displacement'] = np.nan
    for i in range(len(dataset_meal)):
       c_list = dataset_meal.loc[i,:].tolist()
       sum_=[]
       for j in range(1,len(c_list)):
           sum_.append(abs(c_list[j]-c_list[j-1]))
       original_features['Combined Displacement'][i] = np.round(np.sum(sum_),2)
    
    #CGM_Kurtosis
    original_features['Combined Kurtosis'] = np.nan
    for i in range(len(dataset_meal)):
       original_features['Combined Kurtosis'][i] = ts.kurtosis(np.array(dataset_meal.iloc[i,:]))
    
    # Recurring values
    original_features['Combined Recurring values'] = np.nan
    for i in range(len(dataset_meal)):
        original_features['Combined Recurring values'][i] = ts.sum_of_reoccurring_values(dataset_meal.loc[i,:])
    
    
    #Recurr
    original_features['Combined Recur'] = np.nan
    for i in range(len(dataset_meal)):
       original_features['Combined Recur'][i] = ts.ratio_value_number_to_time_series_length(np.array(dataset_meal.iloc[i,:]))
    
    #Remove calculated columns
    del original_features['Combined Max']
    del original_features['Combined Min']
    
    original_features = original_features[['Combined Dist_travelled', 'Combined RMS', 'Combined Velocity', 'Combined MinMax', 'Combined Displacement', 'Combined Recur']]

    return original_features

def Labeling(dataset_meal, dataset_carbs):
    
    label_data = pd.DataFrame(index=np.arange(len(dataset_meal)))

    label_data['Labels'] = 0
    
    for i in range(len(dataset_carbs)):
        
        if  dataset_carbs[0][i] <= 10:
            label_data['Labels'][i] = 0
        elif dataset_carbs[0][i] > 10 and dataset_carbs[0][i] <= 20:
            label_data['Labels'][i] = 1
        elif dataset_carbs[0][i] > 20 and dataset_carbs[0][i] <= 30:
            label_data['Labels'][i] = 2
        elif dataset_carbs[0][i] > 30 and dataset_carbs[0][i] <= 40:
            label_data['Labels'][i] = 3
        elif dataset_carbs[0][i] > 40 and dataset_carbs[0][i] <= 50:
            label_data['Labels'][i] = 4
        elif dataset_carbs[0][i] > 50 and dataset_carbs[0][i] <= 60:
            label_data['Labels'][i] = 5
        elif dataset_carbs[0][i] > 60 and dataset_carbs[0][i] <= 70:
            label_data['Labels'][i] = 6
        elif dataset_carbs[0][i] > 70 and dataset_carbs[0][i] <= 80:
            label_data['Labels'][i] = 7
        elif dataset_carbs[0][i] > 80 and dataset_carbs[0][i] <= 90:
            label_data['Labels'][i] = 8
        elif dataset_carbs[0][i] > 90 and dataset_carbs[0][i] <= 100:
            label_data['Labels'][i] = 9
        else:
            label_data['Labels'][i] = 10
            
    return label_data

def Scaling(X_data):
    
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X_data)

    return X_scaled

def Pca(X_scaled):
    
    pca = PCA(n_components = 2)
    X_pca = pca.fit_transform(X_scaled)

    return X_pca

def k_means(X_pca):
    
    kmeans = KMeans(n_clusters=10, init='k-means++', max_iter=300, n_init=10, random_state=0)
    kmeans.fit(X_pca)
    centroids = kmeans.cluster_centers_

    print(centroids)
    
    
    target = kmeans.fit_predict(X_pca)
    
    plt.figure(figsize=(10, 10))
    colors = ['#e6194B', '#bfef45', '#42d4f4', '#a9a9a9', '#f032e6', '#911eb4', '#4363d8', '#3cb44b', '#ffe119', '#f58231']
    for i in range(10):
        plt.scatter(X_pca[target == i, 0], X_pca[target == i, 1], s = 100, c = colors[i], label = 'Cluster ' + str(i))
    plt.legend()
    plt.show()

#    plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 100, c = 'yellow', label = 'Centroids')

    print("SSE : ", kmeans.inertia_)

    return target

def dbscan(X_pca):
    
    dbscan = DBSCAN(eps = 0.04, min_samples = 3, metric='euclidean')
    cls_db = dbscan.fit_predict(X_pca)
    
    plt.figure(figsize=(10, 10))
    plt.scatter(X_pca[cls_db == 0, 0], X_pca[cls_db == 0, 1], s = 100, c = '#e6194B', label = 'Cluster 1')
    plt.scatter(X_pca[cls_db == 1, 0], X_pca[cls_db == 1, 1], s = 100, c = '#bfef45', label = 'Cluster 2')
    plt.scatter(X_pca[cls_db == 2, 0], X_pca[cls_db == 2, 1], s = 100, c = '#42d4f4', label = 'Cluster 3')
    plt.scatter(X_pca[cls_db == 3, 0], X_pca[cls_db == 3, 1], s = 100, c = '#a9a9a9', label = 'Cluster 4')
    plt.scatter(X_pca[cls_db == 4, 0], X_pca[cls_db == 4, 1], s = 100, c = '#f032e6', label = 'Cluster 5')
    plt.scatter(X_pca[cls_db == 5, 0], X_pca[cls_db == 5, 1], s = 100, c = '#911eb4', label = 'Cluster 6')
    plt.scatter(X_pca[cls_db == 6, 0], X_pca[cls_db == 6, 1], s = 100, c = '#4363d8', label = 'Cluster 7')
    plt.scatter(X_pca[cls_db == 7, 0], X_pca[cls_db == 7, 1], s = 100, c = '#3cb44b', label = 'Cluster 8')
    plt.scatter(X_pca[cls_db == 8, 0], X_pca[cls_db == 8, 1], s = 100, c = '#ffe119', label = 'Cluster 9')
    plt.scatter(X_pca[cls_db == 9, 0], X_pca[cls_db == 9, 1], s = 100, c = '#f58231', label = 'Cluster 10')
    plt.scatter(X_pca[cls_db == 10, 0], X_pca[cls_db == 10, 1], s = 100, c = 'pink', label = 'Cluster 11')

    plt.legend()
    plt.show()

    print("Silhouette Score : ", silhouette_score(X_pca, cls_db))

    return cls_db


def Clustering(feature_data, label_data):
            
    X_scaled = Scaling(feature_data)        
    X_pca = Pca(X_scaled)
    y = np.array(label_data['Labels'])
    
    k_means_labels = k_means(X_pca)
    db_scan_labels = dbscan(X_pca)
    
    print("Correctness k_means:", accuracy_score(y, k_means_labels))
    print("Correctness db_scan:", accuracy_score(y, db_scan_labels))
    
   
if __name__ == "__main__":
    
    dataset_meals, dataset_carbs = preprocess()
    feature_data = feature_evaluation(dataset_meals)
    label_data = Labeling(dataset_meals, dataset_carbs)
    Clustering(feature_data, label_data)