'''
    This script simulates MRG models of axons with DTI-tractography based fiber trajectories.
    The extracellular potential and second spatial difference profiles, as well as the 
    predicted activation initiation site, are plotted and saved as images to a specified output
    directory. The results are also written out to a json so that statistics can be taken.
'''

# Imports
import sys
import numpy as np
import json
import math
import time
import os
from matplotlib import pyplot as plt
from neuron import h, gui

sys.path.append("../")
from lib.DTI import process_DTI
from lib.DTI import graph_DTI
from lib.COMSOL import FEM
from lib.NEURON import axon

h.nrn_load_dll("../lib/NEURON/nrnmech.dll")

# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] 
dti_filename = sys.argv[2] # "../lib/DTI/tracts/L_DRTT_ML_TRP_voxel.txt"
pwd = float(sys.argv[3]) / 1000 # stimulus pulse width, enter in uS
output_image_dir = sys.argv[4]
output_json_dir = sys.argv[5]

diameter = 5.7
STINmul = 10
compartDiv = STINmul + 5
node_to_node = 0.5
num_wfs = 1
frequency = 130

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
fem_bounds = fem.getFEMBounds()

test_dti = process_DTI.DTI_tracts(dti_filename, fem_bounds, node_to_node)

temp_axon = axon.Axon(diameter, 41)
temp_axon_pos = temp_axon.getCompartmentPositions()
all_seg_comp_dists = (np.array(temp_axon_pos)/1000)-0.0005
all_seg_comp_dists = all_seg_comp_dists.round(decimals=5)
single_seg_comp_dists = all_seg_comp_dists[0:compartDiv]

test_dti.getAllComps(single_seg_comp_dists)
xAllComp, yAllComp, zAllComp = test_dti.getAllCompPos()

input_array = []
problem_inds = []

for fib in range(len(xAllComp)):
    # Get compartmental EC potentials from previously made 3d-grid
    fiberLVoltages = []
    for i in range(len(xAllComp[fib])):
        try:
                    
            fiberLVoltages.append(float(grid_e1( [xAllComp[fib][i], yAllComp[fib][i], zAllComp[fib][i]] )))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xAllComp[fib][i]) + ", Y = " + str(yAllComp[fib][i]) + ", Z = " + str(zAllComp[fib][i]))
            pass

    input_array.append(fiberLVoltages)

for i in range(len(input_array)):
    output_image = os.path.join(output_image_dir,"dti_init_" + str(i) + ".jpg")
    axon_temp = axon.Axon(diameter, test_dti.getNodeCount(i))
    temp_axon_pos = temp_axon.getCompartmentPositions()
    all_seg_comp_dists = (np.array(temp_axon_pos)/1000)-0.0005
    all_seg_comp_dists = all_seg_comp_dists.round(decimals=5)

    ecs_all = [k*-1 for k in input_array[i]]

    ecs = [k*-1 for k in input_array[i][::compartDiv]]
    ssds = []
    for j in range(1,len(ecs)-1):
        ssd_temp = ecs[j-1] - (2*ecs[j]) + ecs[j+1]
        ssds.append(ssd_temp)

    ## Use an exponential search to quickly reach threshold without overstimulating
    multiplier = 0.05
    spikes = 0
    while(multiplier < 500):
        spikes, act_inds = axon_temp.stimulate(input_array[i], multiplier, pwd, frequency, num_wfs)
        if spikes < num_wfs:
            multiplier *= 2
        else:
            break
    
    fig = plt.figure()
    node_axis = all_seg_comp_dists[::compartDiv]
    ax = fig.add_subplot(111)
    ax.plot(all_seg_comp_dists, ecs_all, '-b', lw=0.75, ms=1)
    ax.plot(node_axis, ecs, '.b', lw=0.5, ms=6)
    if spikes > 0:
        for ind in act_inds:
            ax.plot(all_seg_comp_dists[ind*compartDiv], ecs_all[ind*compartDiv], '.r', lw=0.75, ms=8)
    title_str = "Threshold = " + str(multiplier) + "V"
    ax.set(title=title_str, xlabel='X (mm)', ylabel='potential (mV)')
    
    ax.grid()
    ax1 = ax.twinx()
    ax1.plot(node_axis[1:len(node_axis)-1],ssds, '.-g', lw=0.75, label='second spatial difference')
    plt.savefig(output_image)
    plt.close()

    output_json = os.path.join(output_json_dir, "dti_init_" + str(i) + ".json")

    tg_dict = {}
    tg_dict["Fiber_Properties"] = {}
    tg_dict["Fiber_Properties"]["diameter"] = diameter
    tg_dict["Fiber_Properties"]["internodal_region_cnt"] = STINmul
    tg_dict["Fiber_Properties"]["dti_index"] = i

    tg_dict["Stimulus_Properties"] = {}
    tg_dict["Stimulus_Properties"]["pulse_width"] = pwd
    tg_dict["Stimulus_Properties"]["num_wfs"] = num_wfs
    tg_dict["Stimulus_Properties"]["frequency"] = frequency

    tg_dict["Activation_Properties"] = {}
    tg_dict["Activation_Properties"]["threshold_multiplier"] = multiplier
    tg_dict["Activation_Properties"]["initiation_nodes"] = act_inds

    tg_dict["Extracellular_Potential_Properties"] = {}
    tg_dict["Extracellular_Potential_Properties"]["ec_potentials"] = ecs


    with open(output_json, 'w') as outfile:
        json.dump(tg_dict, outfile, indent=4)  
      
######################################################################################################