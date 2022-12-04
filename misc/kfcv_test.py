'''
    This is a test script to CORRECTLY implement k fold cross validation for this project.
    Correct implementation requires each fiber containing ALL CASES from a single axonal
    tract. This includes variations in the pulse width and amplitude case. Cross Validation 
    in this way should produce a validation metric that is more representative of ANN 
    performance on unseen fibers than using a single distinct validation set.
'''

import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
import json
import os
import math
import sys

## Comment the below out to use GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

input_dataset = sys.argv[1]

## Specify the feature names
df_column_names = ["Fiber Index", "Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10", "Activation"]

ssd_node_names = ["ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10"]
ec_node_names = ["ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10"]
fsd_node_names = ["fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10"]

dataset = pd.read_csv(
    input_dataset,
    names=df_column_names)

print(dataset.shape)

dataset_features = dataset[["Fiber Index"] + ["Pulse Width"] + ec_node_names + fsd_node_names + ssd_node_names + ["Activation"]]
dataset_labels = dataset_features.pop("Activation")

print(dataset_features.shape)
print(dataset_labels.shape)

def GetDistinctFiberIndices(df):
    # Get number of distinct fiber trajectories
    dti_fiber_inds = []

    for row_ind in range(df.shape[0]):
        if df.iloc[row_ind][0] not in dti_fiber_inds:
            dti_fiber_inds.append(df.iloc[row_ind][0])

    return dti_fiber_inds

dti_fiber_inds = GetDistinctFiberIndices(dataset_features)
print("Number of distinct fiber tracts = " + str(len(dti_fiber_inds)))
    
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)
val_acc_array = []

for train_ind, val_ind in kf.split(dti_fiber_inds):
    features_train = dataset_features[dataset_features["Fiber Index"].isin(np.asarray(dti_fiber_inds)[train_ind])]
    features_val = dataset_features[dataset_features["Fiber Index"].isin(np.asarray(dti_fiber_inds)[val_ind])]
    labels_train = dataset_labels[dataset_features["Fiber Index"].isin(np.asarray(dti_fiber_inds)[train_ind])]
    labels_val = dataset_labels[dataset_features["Fiber Index"].isin(np.asarray(dti_fiber_inds)[val_ind])]

    #features_train.pop("Fiber Index")
    #features_val.pop("Fiber Index")

    dti_fiber_inds_train = GetDistinctFiberIndices(features_train)
    print("Number of distinct training fiber tracts = " + str(len(dti_fiber_inds_train)))

    dti_fiber_inds_val = GetDistinctFiberIndices(features_val)
    print("Number of distinct val fiber tracts = " + str(len(dti_fiber_inds_val)))

    print(features_train.shape)
    print(features_val.shape)
    print(labels_train.shape)
    print(labels_val.shape)

    #print(features_train.shape)
    #print(np.asarray(dti_fiber_inds)[val_ind])
    #print(labels_train.shape)
    #print(labels_val.shape)

    print()