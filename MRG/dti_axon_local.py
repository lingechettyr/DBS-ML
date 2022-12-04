'''
    This script predicts the activation of the DTI-tractography based fiber trajectories
    using the MRG compartment model. The results are graphed with mayavi mlab.
'''

# Imports
import sys
import numpy as np
import json
import math
import time
from neuron import h, gui

sys.path.append("../")
from lib.DTI import process_DTI
from lib.DTI import graph_DTI
from lib.COMSOL import FEM
from lib.NEURON import axon

h.nrn_load_dll("../lib/NEURON/nrnmech.dll")

# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] 
dti_filename = sys.argv[2] #"../lib/DTI/tracts/L_DRTT_ML_TRP_voxel.txt"

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

xNodeComp, yNodeComp, zNodeComp = test_dti.getNodeCompPos()
test_dti_graph = graph_DTI.DTI_grapher(test_dti.getLeadCoordinates(), xNodeComp, yNodeComp, zNodeComp)

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
                    
            fiberLVoltages.append((grid_e1( [xAllComp[fib][i], yAllComp[fib][i], zAllComp[fib][i]] )))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xAllComp[fib][i]) + ", Y = " + str(yAllComp[fib][i]) + ", Z = " + str(zAllComp[fib][i]))
            pass

    input_array.append(fiberLVoltages)

while(True):

    pwd = float(input("Enter the stimulus pulse width in us: ")) / 1000
    multiplier = float(input("Enter the stimulus multiplier: "))

    act_inds = []

    prediction_start_time = time.time()
    for i in range(len(input_array)):
        axon_temp = axon.Axon(diameter, test_dti.getNodeCount(i))
        spikes, _ = axon_temp.stimulate(input_array[i], multiplier, pwd, frequency, num_wfs)
        if spikes >= num_wfs:
            act_inds.append(i)
    
    prediction_stop_time = time.time()
    print()
    print(str(len(input_array)) + " ANN predictions took " + str(prediction_stop_time - prediction_start_time) + " s")
    print()
    
    test_dti_graph.plotActivatedTracts(act_inds, problem_inds)
      
######################################################################################################