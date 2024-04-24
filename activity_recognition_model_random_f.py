# -*- coding: utf-8 -*-
"""
Date        : Created on Thu Oct  5 12:20:01 2023
Author      : Lizeth Gonzalez Carabarin
Description : This file trains and tests a model to categorize 6 different activities
based on the measures on accelerometers placed at ankles and writs of multiple
subjects. The dataset can be found at:
https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/
This file train the Random Forrest model to categorize over only one individual.

TODO:
- Train using the XGBoost    
- Train over the full dataset (meaning all subjects)

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, ConfusionMatrixDisplay
from sklearn.model_selection import RandomizedSearchCV, train_test_split



##############################################################################

# STEP 1: Download the data from dataset

#Reading data
# TODO : place here your own path
path = './data/raw_accelerometry_data/id00b70b13.csv'


def read_data(path):
    """This functions reads data from a cvs file and converts it into a numpy
       array """
    df_data = pd.read_csv(path)
    dataset = df_data.to_numpy()
    return dataset

# create a dataset
dataset = read_data(path)

# separate the dataset by X data and Y data. Y_data will be a numerical value
# that categorize certain activity, while X_data contains the accelerometer data
X_data = dataset[:,1:-1];
Y_data = dataset[:,0]; 


# count how many actual categories (activities) are
unique, counts = np.unique(Y_data, return_counts=True)
print("category numbers are :", unique)

"""
https://physionet.org/content/accelerometry-walk-climb-drive/1.0.0/raw_accelerometry_data_dict.csv
Correspodance of numerical value with an activity
1  : Walking
2  : Descending stairs
3  : Ascending stairs
4  : Driving
77 : Clapping
99 : Others

"""

## This just converts all 77's to 5's and all 99's to 6's just to have
## some consistency
Y_data[Y_data == 77] = 5
Y_data[Y_data == 99] = 6

"""
Final correspodance of numerical value with an activity
1  : Walking
2  : Descending stairs
3  : Ascending stairs
4  : Driving
5  : Clapping
6  : Others

"""

unique, counts = np.unique(Y_data, return_counts=True)
print("category numbers are :", unique)

#separate training data from testing data
# You use training data to train your model and test data to test your already
# trained model
X_train, X_test, y_train, y_test = train_test_split(X_data, Y_data, test_size=0.2)

#Training the actual model (random forrest)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
# TODO: Train and test also using XGBoost model

#test your model
y_pred = rf.predict(X_test)

# get accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

#TODO Look for other estimators (precision, recall etc)
