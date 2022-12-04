'''
    This script is used to evaluate a single combination of hyperparameters. The structure
    of this script and the supporting scripts (csv generator and ANN training library) 
    support the parallelization of the hyperparameter search across HiPerGator.
'''

import numpy as np
import pandas as pd
import sys
import ann_train_lib
import csv
from torch.utils.tensorboard import SummaryWriter

hparam_csv_name = sys.argv[1]
csv_row_ind = int(sys.argv[2])
regression = 0

num_trials = 3

## Manually list the activation functions, this is weird and should be improved
act_funcs = ["relu", "tanh", "sigmoid"]

## Manually list the datasets, this is weird and should be improved
datasets = ["datasets/7-5/train_orth/class/ds_orth_pw_1_ec_train_20_100_10.csv"]
val_datasets = ["datasets/7-5/val_orth/class/ds_orth_pw_1_ec_train_20_100_10.csv"]

# CSV Input Params
with open(hparam_csv_name) as fd:
    csv_reader = csv.reader(fd)
    row = [row for idx, row in enumerate(csv_reader) if idx == csv_row_ind]

# Specify hyperparameters
num_ecs = int(row[0][0])
num_fsds = int(row[0][1])
num_ssds = int(row[0][2])
num_layers= int(row[0][3])
neurons= int(row[0][4])
dropout= float(row[0][5])
act_func= row[0][6]
l_rate= float(row[0][7])
epochs= int(row[0][8])
batch_size= int(row[0][9])
train_ds = row[0][10]
val_ds = row[0][11]

act_func_mod = act_funcs.index(act_func)
train_ds_mod = datasets.index(train_ds)

val_ds = val_datasets[train_ds_mod]

temp_ANN = ann_train_lib.ANN_train(regression, num_ecs, num_fsds, num_ssds, num_layers, neurons, dropout, act_func, l_rate, epochs, batch_size)
train_accs = []
val_accs = []

for ind in range(num_trials):
    temp_ANN.Build()
    train_acc_temp, val_acc_temp = temp_ANN.TrainEarlyStoppage(train_ds, val_ds)

    if regression == 1 and train_acc_temp != 100:
        train_accs.append(train_acc_temp)
        val_accs.append(val_acc_temp)
    elif regression == 0:
        train_accs.append(train_acc_temp)
        val_accs.append(val_acc_temp)

if len(train_accs) > 0:
    train_avg_acc = np.mean(np.asarray(train_accs))
    val_avg_acc = np.mean(np.asarray(val_accs))
else:
    train_avg_acc = 100
    val_avg_acc = 100

print("Training accuracy: " + str(train_avg_acc))
print("Validation accuracy (MAPE): " + str(val_avg_acc))

exp = SummaryWriter()
exp.add_hparams(
    {"num_volts": num_ecs, "num_fsds": num_fsds, "num_ssds": num_ssds, "hidden_layers": num_layers, "neurons": neurons, "act_func": act_func_mod, "dropout": dropout, "l_rate": l_rate, "epochs": epochs, "batch_size": batch_size,  "train_ds": train_ds_mod},
    {
        "Training Accuracy": train_avg_acc,
        "Validation Accuracy": val_avg_acc
    }
)

exp.close()