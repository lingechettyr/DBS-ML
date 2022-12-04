''' 
    This script is used to setup random hyperparameter searches for optimizing the 
    ANN models. The script creates a csv where each row is a random combination of
    hyperparameters, based on the specified values/ranges.
'''

import numpy as np
import csv
import sys
import random

output_file = sys.argv[1]
num_rows = int(sys.argv[2])

class HParam:
    def __init__(self, name, discrete_list, start, stop, val_type):
        self.name = name
        self.discrete_list = discrete_list
        self.start = start
        self.stop = stop
        self.val_type = val_type

    def GetRandomValue(self):
        if self.discrete_list != None:
            return random.choice(self.discrete_list)
        elif self.val_type == "int":
            return random.randrange(self.start, self.stop+1)

#### Parameter staging ####
hparams_list = list()

hparams_list.append(HParam('num_ecs', [11], None, None, None))
hparams_list.append(HParam('num_fsds', [11], None, None, None))
hparams_list.append(HParam('num_ssds', [11], None, None, None))
hparams_list.append(HParam('num_layers', None, 1, 4, 'int'))
hparams_list.append(HParam('neurons', None, 15, 500, "int"))
hparams_list.append(HParam('dropout', [0.0, 0.2, 0.4], None, None, None))
hparams_list.append(HParam('act_func', ["relu","tanh", "sigmoid"], None, None, None))
hparams_list.append(HParam('l_rate', [0.0001, 0.001, 0.01], None, None, None))
#hparams_list.append(HParam('epochs', [None, 1, 50], "int")) # this not needed if using early stopping
hparams_list.append(HParam('epochs', [100], None, None, None))
hparams_list.append(HParam('batch_size', [32, 64, 128], None, None, None))
hparams_list.append(HParam('train_ds', ["datasets/7-5/train_orth/class/ds_orth_pw_1_ec_train_20_100_10.csv"], 
                                        None, None, None))
hparams_list.append(HParam('val_ds', [None], None, None, None))


#### Create rows of independent hparam combinations ####
hparam_rows = []
for i in range(num_rows):
    unique_combination = False
    while(unique_combination == False):
        unique_combination = True
        hparams_temp = []
        for j in range(len(hparams_list)):
            hparams_temp.append(hparams_list[j].GetRandomValue())

        for row in hparam_rows:
            if row == hparams_temp:
                unique_combination = False
                break

    hparam_rows.append(hparams_temp)

f = open(output_file, 'w+', newline='')
writer = csv.writer(f)

hparam_names = [p.name for p in hparams_list]
writer.writerow(hparam_names)
for row in hparam_rows:
    writer.writerow(row)