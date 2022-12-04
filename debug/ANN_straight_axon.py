'''
    This script allows the user to make predictions on individual fibers of specified orientation, 
    position, pulse width, amplitude, and node alignment. Both the MRG and ANN models are ran, so
    that their outputs can be compared.
'''

# Imports
import sys
import numpy as np
import time
from neuron import h, gui
import json
import matplotlib.pyplot as plt
import math
import time
from random import random

sys.path.append("../")
from lib.NEURON import axon
from lib.COMSOL import FEM
from ANN import ann_predict_lib

h.nrn_load_dll("../lib/NEURON/nrnmech.dll")

# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] 
fiber_h_distance = float(sys.argv[2]) # the horizontal between the electrode lead and the fiber, in mm
fiber_v_displacement = float(sys.argv[3]) # the vertical displacement of the fiber center from the lead plane, in mm
number_of_nodes = int(sys.argv[4])
para = int(sys.argv[5])
ANN_model = sys.argv[6]

SSD_centered = 1

showGraph = 1
diameter = 5.7
STINmul = 10
compartDiv = STINmul + 5
node_to_node = 0.5
num_wfs = 1
frequency = 130

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()

axon = axon.Axon(diameter,number_of_nodes)
compartmentPos = np.asarray(axon.getCompartmentPositions())

ANN_model = ann_predict_lib.ANN(ANN_model)
num_ecs, num_fsds, num_ssds = ANN_model.get_input_sizes()

orthogonal_offset = 0 # center of the active contact in the X dimension
parallel_offset = 6.75 # center of the active contact in the Z dimension

if para == 0:
    used_offset = orthogonal_offset
else:
    used_offset = parallel_offset

pos_arry = (compartmentPos/1000)+used_offset-(((number_of_nodes-1)/2)*node_to_node)-0.0005
pos_arry = pos_arry.round(decimals=5)
pos_arry = pos_arry[0:int((number_of_nodes-1)*compartDiv)+1]
fiberLVoltages = np.zeros(len(pos_arry), dtype=float)  #container for voltage data at nodes only    

init_mul = 0

while(True):
    shift = float(input("Enter the fiber shift: ")) 
    pwd = float(input("Enter the pulse width in uS: ")) / 1000
    multiplier = float(input("Enter the EC multiplier (If predicting thresholds, enter '1'): ")) 

    if para == 0:
        selectedFiberLx = pos_arry - (shift * node_to_node)
        selectedFiberLy = np.ones(len(selectedFiberLx)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLz = np.ones(len(selectedFiberLx)) * (6.75 + fiber_v_displacement)    #specifying nodal locations for a fiber
    else:
        selectedFiberLz = pos_arry - (shift * node_to_node)
        selectedFiberLy = np.ones(len(selectedFiberLz)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLx = np.ones(len(selectedFiberLz)) * 0    #specifying nodal locations for a fiber

    # Get compartmental EC potentials from previously made 3d-grid
    for i in range(len(selectedFiberLx)):
        try:
                    
            fiberLVoltages[i]=grid_e1( [selectedFiberLx[i], selectedFiberLy[i], selectedFiberLz[i]] )
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range!")
            pass
                
    voltages1 = fiberLVoltages

    #spikes, _ = axon.stimulate(voltages1, multiplier, pwd, 1000, 1)
    threshold = axon.findThreshold(voltages1, pwd, 1000, 1)
    print()
    print("MRG prediction: " + str(threshold))
    print()

    fiberLVoltages_nodes = fiberLVoltages[::15]

    ssds = []
    ecs = []
    fsds = []

    center_ind = 0

    if SSD_centered == 1:
        max_ssd_ind = 0
        max_ssd = 0
        for ind in range(1,len(fiberLVoltages_nodes)-1):
            temp_ssd = -1 * (fiberLVoltages_nodes[ind-1] - (2*fiberLVoltages_nodes[ind]) + fiberLVoltages_nodes[ind+1])
            if temp_ssd > max_ssd:
                max_ssd = temp_ssd
                max_ssd_ind = ind

        center_ind = max_ssd_ind
    else:
        max_ec_ind = 0
        max_ec = 0
        for ind in range(1,len(fiberLVoltages_nodes)-1):
            temp_ec = fiberLVoltages_nodes[ind]
            if temp_ec > max_ec:
                max_ec = temp_ec
                max_ec_ind = ind

        center_ind = max_ec_ind

    for ind in range(center_ind - (math.floor(num_ecs / 2)), center_ind + (math.floor(num_ecs / 2)) + 1):
        temp_ec = -1 * fiberLVoltages_nodes[ind] # stimulus is cathodic, so multiply by -1
        ecs.append(temp_ec)

    for ind in range(center_ind - (math.floor(num_fsds / 2)), center_ind + (math.floor(num_fsds / 2)) + 1):
        temp_fsd = abs(fiberLVoltages_nodes[ind+1] - fiberLVoltages_nodes[ind-1]) / 2 
        fsds.append(temp_fsd)

    for ind in range(center_ind - (math.floor(num_ssds / 2)), center_ind + (math.floor(num_ssds / 2)) + 1):
        temp_ssd = -1 * (fiberLVoltages_nodes[ind-1] - (2*fiberLVoltages_nodes[ind]) + fiberLVoltages_nodes[ind+1]) # stimulus is cathodic, so multiply by -1
        ssds.append(temp_ssd)

    if num_ecs == 0:
        ecs.pop()
    if num_fsds == 0:
        fsds.pop()
    if num_ssds == 0:
        ssds.pop()

    input_tensor = [pwd]
    input_tensor.extend(np.multiply(ecs, multiplier))
    input_tensor.extend(np.multiply(fsds, multiplier))
    input_tensor.extend(np.multiply(ssds, multiplier))

    ANN_prediction = ANN_model.predict_threshold(input_tensor, 50)
    print()
    print("ANN prediction: " + str(ANN_prediction))
    print()

    if showGraph == 1:
        secSpatialDiffs = list()
        for i in range(1,len(fiberLVoltages_nodes)-1):
            temp_ssd = (fiberLVoltages_nodes[i-1] - (2*fiberLVoltages_nodes[i]) + fiberLVoltages_nodes[i+1])
            secSpatialDiffs.append(temp_ssd)

        figAllGood = plt.figure()
        x_axis = range(len(fiberLVoltages_nodes))
        ax = figAllGood.add_subplot()
        ax.plot(x_axis,((np.array(fiberLVoltages_nodes))),'.-b', lw=1, ms=4)
        
        ax.grid()
        ax1 = ax.twinx()
        ax1.plot(x_axis[1:len(x_axis)-1],secSpatialDiffs, '.-g', lw=1, ms=4, label='second spatial difference')
        plt.show()
             
######################################################################################################