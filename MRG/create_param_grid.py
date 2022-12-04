''' 
    This file generates a grid-search of specified parameters in csv format.
    Run the script with 'python create_param_csv.py <output csv filename>'
    Parameters can be added to the search with the following line:
    params_list.append(Param('<name>',<array of values>))
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import ParameterGrid
import csv
import sys
import math

output_file = sys.argv[1]

class Param:
    def __init__(self, name, vals):
        self.name = name
        self.vals = vals

params_list = list()

#### Parameter staging ####

## The below for straight fibers on hpg
#params_list.append(Param('h_dist', list(np.arange(400,6400+1,400))))
params_list.append(Param('h_dist_start', list([500])))
params_list.append(Param('h_dist_stop', list([12500])))
params_list.append(Param('h_dist_step', list([800])))
params_list.append(Param('v_disp', list(np.arange(-12000,12000+1,800)))) # for orthogonal fibers
#params_list.append(Param('v_disp', list([0]))) # for parallel fibers
params_list.append(Param('pw', list(np.arange(30,150+1,15)) + list([175,200,225,250,275,300,350,400,450,500])))

## The below for DTI fibers on hpg
# num_of_tracts = 100
# tracts_per_node = 50

# params_list.append(Param('tract_cnt', list([tracts_per_node])))
# params_list.append(Param('tract_ind', range(math.ceil(num_of_tracts/tracts_per_node))))
# params_list.append(Param('pw', list(np.arange(30,150+1,15)) + list([175,200,225,250,275,300,350,400,450,500])))

###########################

param_grid = {p.name:p.vals for p in params_list}
grid = ParameterGrid(param_grid)

f = open(output_file, 'w+', newline='')
writer = csv.writer(f)

param_names = [p.name for p in params_list]
writer.writerow(param_names)
for params in grid:
    row = [params[p.name] for p in params_list]
    writer.writerow(row)