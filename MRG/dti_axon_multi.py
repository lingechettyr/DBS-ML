'''
    This script predicts the activation of the DTI-tractography based fiber trajectories
    using the MRG compartment model. The thresholds are outputted to json files, 
    one for each fiber stimulated. The fiber index and pulse widths to 
    stimulate at are inputted via a row from a csv. This script can also be used for 
    other fiber trajectories, as long as they are given in the same format as the DTI
    trajectories.
'''

# Imports
import sys
import numpy as np
import math
import time
import json
import csv
import os
from neuron import h, gui

sys.path.append("../")
from lib.DTI import process_DTI
from lib.COMSOL import FEM
from lib.NEURON import axon

h.nrn_load_dll("../lib/NEURON/x86_64/.libs/libnrnmech.so")

# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] 
tractFile = sys.argv[2]
output_dir = sys.argv[3] 
df_csv = sys.argv[4]
df_index = int(sys.argv[5])

diameter = 5.7
STINmul = 10
compartDiv = STINmul + 5
node_to_node = 0.5
num_wfs = 1
frequency = 130

# CSV Input Params
with open(df_csv) as fd:
    csv_reader = csv.reader(fd)
    row = [row for idx, row in enumerate(csv_reader) if idx == df_index]

tract_cnt = int(row[0][0])
tract_ind = int(row[0][1])
pulse_width_str = row[0][2]

tract_sim_inds = range(tract_cnt*tract_ind, (tract_cnt*tract_ind) + tract_cnt)
pwd = float(pulse_width_str) / 1000

def mkdirp(dir):
    '''make a directory if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

mkdirp(output_dir)        

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
fem_bounds = fem.getFEMBounds()

test_dti = process_DTI.DTI_tracts(tractFile, fem_bounds, node_to_node)

temp_axon = axon.Axon(diameter, 41)
temp_axon_pos = temp_axon.getCompartmentPositions()
all_seg_comp_dists = (np.array(temp_axon_pos)/1000)-0.0005
all_seg_comp_dists = all_seg_comp_dists.round(decimals=5)
single_seg_comp_dists = all_seg_comp_dists[0:compartDiv]

test_dti.getAllComps(single_seg_comp_dists)
xAllComp, yAllComp, zAllComp = test_dti.getAllCompPos()

input_array = []
problem_inds = []

for fib in tract_sim_inds:
    if fib not in range(len(xAllComp)): # make sure not an edge case
        break

    # Get compartmental EC potentials from previously made 3d-grid
    fiberLVoltages = []
    for i in range(len(xAllComp[fib])):
        try:
                    
            fiberLVoltages.append(float(grid_e1( [xAllComp[fib][i], yAllComp[fib][i], zAllComp[fib][i]] )))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xAllComp[fib][i]) + ", Y = " + str(yAllComp[fib][i]) + ", Z = " + str(zAllComp[fib][i]))
            pass

    input_array.append(fiberLVoltages)

itr = 0
for fib in tract_sim_inds:
    if fib not in range(len(xAllComp)): # make sure not an edge case
        break

    output_file = output_dir + "/DTI_" + str(fib) + "_" + str(pulse_width_str) + ".json"

    ## Comment out below if you wish to overwrite existing sim jsons of the same name ##
    if os.path.exists(output_file):
        itr += 1
        continue

    axon_temp = axon.Axon(diameter, test_dti.getNodeCount(fib))
    threshold = axon_temp.findThreshold(input_array[itr], pwd, frequency, num_wfs)
    voltages = [k*-1 for k in input_array[itr][::compartDiv]]
    
    tg_dict = {}
    tg_dict["Fiber_Properties"] = {}
    tg_dict["Fiber_Properties"]["diameter"] = diameter
    tg_dict["Fiber_Properties"]["internodal_region_cnt"] = STINmul
    tg_dict["Fiber_Properties"]["dti_index"] = fib

    tg_dict["Stimulus_Properties"] = {}
    tg_dict["Stimulus_Properties"]["pulse_width"] = pwd
    tg_dict["Stimulus_Properties"]["num_wfs"] = num_wfs
    tg_dict["Stimulus_Properties"]["frequency"] = frequency

    tg_dict["Activation_Properties"] = {}
    tg_dict["Activation_Properties"]["threshold_multiplier"] = threshold

    tg_dict["Extracellular_Potential_Properties"] = {}
    tg_dict["Extracellular_Potential_Properties"]["ec_potentials"] = voltages

    with open(output_file, 'w') as outfile:
        json.dump(tg_dict, outfile, indent=4)     

    itr += 1   
    
######################################################################################################