Group: 34

Members:

Manish Aakaram	 1217852896	maakaram@asu.edu
Karishma Joseph	 1217137207	kjoseph7@asu.edu
Gowtham Sekkilar 1215181396	gsekkila@asu.edu
Jerold Thomas	 1215139965	jjthom25@asu.edu

-------------------------------------------------------------------------------------------------------------------------------
Classifiers :

Support Vector Machine(SVM) has been done by Jerold Thomas
Multi-Layer Perceptron(MLP) has been done by Karishman Joseph
Random Forest Classifier(RForest) has been done by Gowtham Sekkilar
Logistic Regression(LOGR) has been done by Manish Aakaram

We also worked together on two other classifiers Adaboost and KNN.
-------------------------------------------------------------------------------------------------------------------------------
The contents of the zip are:

Readme File

K_Fold File.py 

--> This contains the K-fold cross validation for both meal and no meal data.
--> This file also generates the values for the metrics Accuracy, F-1 Score, Recall and Precision.

Training.py 

--> This file contains the trained model of all the classifiers for the given meal and no meal data. 
--> It generates the classifier model files and those model files are stored in the same folder.

Following are the generated classifier model files:

svm_model.pkl
mlp_model.pkl
rforest_model.pkl
logr_model.pkl
adaboost_model.pkl
knn_model.pkl
scaling.pkl
pca.pkl

Testing.py

--> Predicts class label based on the trained models for test data and outputs the labels into csv files.

EigenVector.csv

--> Contains the generated Eigen vectors from the trained data. 
-------------------------------------------------------------------------------------------------------------------------------
How to Run the files? (Only in Windows)

(All the contents of the zip file should be stored inside "....\Assingment 2 - Group 34")

--> Extract the zip folder 'Assignment 2 - Group 34' in the current working directory of the system.  

--> Open Anaconda prompt in windows and change to the folder directory:

cd Assignment 2 - Group 34

--> To see the accuracy, f1-score, recall and precision - Run the K_Fold.py using the following command:

python K_Fold.py

--> To run the Testing.py file, use the following command:

python Testing.py your_file_name.csv

CSV Files with labels will be generated in the current folder.

If running the files in MacOS then the directory path of the file should be changed in K_Fold, Training and Testing to the respective file locations.





