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
import ann_predict_lib


model_name = sys.argv[1] # path/name of model
dataset_path = sys.argv[2]



predict_ANN = ann_predict_lib.ANN(model_name)

dataset_features = ["Pulse Width", "ec_0", "ec_1", "ec_2", "ec_3", "ec_4", "ec_5", "ec_6", "ec_7", "ec_8", "ec_9", "ec_10", "fsd_0", "fsd_1", "fsd_2", "fsd_3", "fsd_4", "fsd_5", "fsd_6", "fsd_7", "fsd_8", "fsd_9", "fsd_10", "ssd_0", "ssd_1", "ssd_2", "ssd_3", "ssd_4", "ssd_5", "ssd_6", "ssd_7", "ssd_8", "ssd_9", "ssd_10"]

dataset = pd.read_csv(
            dataset_path,
            names=dataset_features)



predictions = predict_ANN.predict(dataset)	
