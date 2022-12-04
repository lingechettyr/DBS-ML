'''
    This script creates and trains a single ANN model on specified hyperparameters. The
    model is stored as a tensorflow object to the 'saved_models' directory.
'''

import numpy as np
import pandas as pd
import sys
import os
import modified_ann_train_lib

model_name = sys.argv[1] # path/name of model, also where to create save and normalization files
train_path = sys.argv[2]
val_path = sys.argv[3]

print('Here')
print(val_path)

# Specify hyperparameters
regression = 0
num_ecs = 11
num_fsds = 11
num_ssds = 11
num_layers = 20
neurons = 46
dropout = 0.0
act_func = "relu"
l_rate = 0.001
epochs = 300 #28
batch_size = 32

train_ds = train_path
val_ds = val_path

temp_ANN = modified_ann_train_lib.ANN_train(regression, num_ecs, num_fsds, num_ssds, num_layers, neurons, dropout, act_func, l_rate, epochs, batch_size)
temp_ANN.Build()
train_acc, val_acc = temp_ANN.TrainEarlyStoppage(train_ds, val_ds)

print("Training Accuracy = " + str(train_acc))
print("Validation Accuracy (MAPE) = " + str(val_acc))

temp_ANN.SaveModel(model_name)
