'''
    This script is used to parse all json output files from HPG,
    formatting the relevant data into rows to append to a dataset csv.
    Multiple independent datasets can be defined using the relevant 
    parameters, and the json files will be included only in the  
    appropriate csv(s) to which they apply. The script is currently 
    setup to run from an array-job on HPG, and takes an array index
    that it uses to select specific datasets.

    Additions:
    6/30: Datasets can now be "stratified" by threshold amplitude
          to ensure that each dataset has a uniform distribution
          of thresholds.
'''

import sys
import json
import csv
import numpy as np
from pathlib import Path
import pandas as pd
import sys
import os
import math

input_path = sys.argv[1] # name of input json path
output_path = sys.argv[2] # name of output csv path
array_index = int(sys.argv[3]) # SLURM array index

number_of_spatial_values = 11 # must be an odd number

# A class containing the relevant parameter and csv information for single
# dataset (csv file) to be generated.
class Dataset:
    pulse_widths = []
    h_dists = []
    v_disps = []
    
    def __init__(self, fib_type, hds, vds, pws, cnt, ds_type, dist_sd, dist_size, amp_cutoff, strata_height, fn):
        self.fib_type = fib_type
        self.h_dists = hds
        self.v_disps = vds
        self.pulse_widths = pws
        self.center = cnt
        self.ds_type = ds_type
        self.dist_sd = dist_sd
        self.dist_size = dist_size
        self.amp_cutoff = amp_cutoff
        self.strata_height = strata_height
        self.csv_filename = fn

# Create an empty array of datasets, add specific instances as needed. 
datasets = []

### Create specific instances of datasets with desired parameters (as many as desired, although keep in mind all must be generated from the same json result directory on HPG)####
datasets.append(Dataset("dti", None, None, list(np.arange(60,150+1,30)) + list([200,250,300,400,500]), 'ec', "train", .20, 100, 10, None, 'ds_dti_pw_1_ec_train_20_100_10.csv'))

# Get single dataset from dataset array using SLURM array index
ds = datasets[array_index-1]  
data = []

good_cnt = 0
bad_cnt = 0

strata_array = np.zeros(int(ds.amp_cutoff))

# Check all files in input path, can this be improved somehow?
for filename in sorted(Path(input_path).rglob("*.json")):
    json_filename = os.path.basename(filename)
    splits = json_filename.split('_')

    # If the filename parameters belong to the current dataset, proceed with training example generation.
    if ds.fib_type == "straight" and int(splits[1]) in ds.h_dists and int(splits[2]) in ds.v_disps and int(splits[3].split('.')[0]) in ds.pulse_widths:
        with open(filename) as f:
            preData = json.load(f)
    elif ds.fib_type == "dti" and int(splits[2].split('.')[0]) in ds.pulse_widths:
        with open(filename) as f:
            preData = json.load(f)
    else:
        continue

    # Get some relevant data from the jsons
    pulse_width = preData["Stimulus_Properties"]["pulse_width"]
    threshold = preData['Activation_Properties']['threshold_multiplier']

    # Don't want to train/validate/test on Vths over a certain amplitude
    if threshold > int(ds.amp_cutoff):
        bad_cnt += 1
        continue

    if ds.strata_height != None:
        strata_index = int(math.floor(threshold))
        if strata_array[strata_index] > ds.strata_height:
            continue
        else:
            strata_array[strata_index] += 1
    
    good_cnt += 1

    voltages_nodes = preData['Extracellular_Potential_Properties']['ec_potentials']

    max_ec = 20
    max_ec_node = 0          
    max_ssd = -20
    max_ssd_node = 0
    for i in range(1,len(voltages_nodes)-1):
        temp_ssd = voltages_nodes[i-1] - (2*voltages_nodes[i]) + voltages_nodes[i+1]
        if temp_ssd > max_ssd:
            max_ssd = temp_ssd
            max_ssd_node = i

        if voltages_nodes[i] < max_ec:
            max_ec = voltages_nodes[i]
            max_ec_node = i

    if ds.center == 'ec':
        max_ssd_node = max_ec_node
    elif ds.center == 'ssd':                                                                   
        max_ec_node = max_ssd_node

    ecs = []
    fsds = []
    ssds = []

    input_bound = math.floor(number_of_spatial_values / 2) + 1

    if max_ssd_node < input_bound or len(voltages_nodes) - 1 - max_ssd_node < input_bound:
        continue
    if max_ec_node < input_bound or len(voltages_nodes) - 1 - max_ec_node < input_bound:
        continue

    for i in range(number_of_spatial_values):
        offset = int((number_of_spatial_values - 1) / 2)

        ecs.append(-1 * voltages_nodes[max_ec_node + i - offset]) # want to take the negative of the voltages, which will result in positive values as they are generated from a cathodic source
        fsds.append((voltages_nodes[max_ec_node + i - offset + 1] - voltages_nodes[max_ec_node + i - offset - 1]) / 2)
        ssds.append(voltages_nodes[max_ssd_node + i - offset - 1] - (2 * voltages_nodes[max_ssd_node + i - offset]) + voltages_nodes[max_ssd_node + i - offset + 1])

    if ds.ds_type == "train":
        # Generate a distribution of multipliers (amplitudes) around the computed threshold.
        # Use this distribution to create a number of binary-activation examples.
        # This is allowed because both extracellular potentials (ECs) and their second-spatial-differences (SSDS)
        # can be linearly scaled.

        amps = np.random.normal(threshold, threshold * ds.dist_sd, ds.dist_size)
    
        for i in range(len(amps)):
            row = []
            row.append(pulse_width)

            for j in range(number_of_spatial_values):
                row.append(ecs[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(fsds[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(ssds[j] * amps[i])
            
            if amps[i] < threshold:
                row.append(0)
            else:
                row.append(1)

            data.append(row)

    elif ds.ds_type == "val":
        # Alternatively, generate just two close binary activation examples, one on 
        # each side of the threshold. This is typically used for the validation
        # datasets when redundancy isn't neccessary.

        diff = threshold * ds.dist_sd 
        amps = [threshold - diff, threshold + diff]

        for i in range(len(amps)):
            row = []
            row.append(pulse_width)

            for j in range(number_of_spatial_values):
                row.append(ecs[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(fsds[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(ssds[j] * amps[i])
            
            if amps[i] < threshold:
                row.append(0)
            else:
                row.append(1)

            data.append(row)

    elif ds.ds_type == "train_reg":
        # This option is to train the ANN to predict not binary activation, but the
        # activation threshold multiplier itself.
 
        amps = [1]

        for i in range(len(amps)):
            row = []
            row.append(pulse_width)

            for j in range(number_of_spatial_values):
                row.append(ecs[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(fsds[j] * amps[i])

            for j in range(number_of_spatial_values):
                row.append(ssds[j] * amps[i])
            
            row.append(threshold)

            data.append(row)

print(good_cnt)
print(bad_cnt)

# Dump the created dataframe to the specified csv file.
df = pd.DataFrame(data)
df.to_csv(os.path.join(output_path, ds.csv_filename), header=False, index=False)