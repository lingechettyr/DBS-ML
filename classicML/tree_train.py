import numpy as np
import pandas as pd
import sys
import os
import tree_train_lib

model_name = sys.argv[1] # path/name of model, also where to create save and normalization files

# Specify hyperparameters
regression = 0
num_ecs = 0
num_fsds = 11
num_ssds = 11
eta = 0.2
max_depth = 8
tree_method = "hist"
objective = "binary:logistic"
eval_metric = "error"

dataset = "ds_dti_pw_1_ec_train_20_100_10.csv"

temp_tree = tree_train_lib.Tree_train(regression, num_ecs, num_fsds, num_ssds, eta, max_depth, tree_method, objective, eval_metric)
temp_tree.build()
train_acc, val_acc = temp_tree.trainEval(dataset)

print("Training Accuracy = " + str(train_acc))
print("Validation Accuracy = " + str(val_acc))

temp_tree.featureImportance()
temp_tree.crossValBoxPlot(dataset)
temp_tree.graphTree()

temp_tree.SaveModel(model_name)

