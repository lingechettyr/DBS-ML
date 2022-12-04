'''
    This script plots the voltage and second derivative profiles of straight
    fibers at specified locations and orientation. It is useful for identifying
    problems in the FEM and regions where the ANN may have difficulty.
'''

# Imports
import sys
import numpy as np
import time
import json
import matplotlib.pyplot as plt

sys.path.append("../")
from lib.COMSOL import FEM

# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] #"V-3d-1V-monopolar.txt" # E-field file from first electrode

# Create interpolated grid from FEM export
fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
print("FEM processing complete")
print()

# Define number of nodes and internodal distance
number_of_nodes = 81
node_to_node = 0.5

x = np.arange(0-(((number_of_nodes-1)/2)*node_to_node), 0+(((number_of_nodes-1)/2)*node_to_node)+node_to_node, node_to_node)
z = np.arange(6.75-(((number_of_nodes-1)/2)*node_to_node), 6.75+(((number_of_nodes-1)/2)*node_to_node)+node_to_node, node_to_node)

# print(x)
# print()
# print(z)
# print()

fiberLVoltages = np.zeros(len(x), dtype=float)  #container for voltage data at nodes only    

init_mul = 0

while(True):
    para = int(input("Enter the fiber orientation to the DBS lead (0 - orthgonal, 1 - parallel): "))
    fiber_h_distance = float(input("Enter the distance (mm) of the fiber from the surface of the lead: ")) 
    fiber_v_displacement = float(input("Enter the displacement (mm) of the fiber from the center of the active lead (z = 6.75 mm): "))
    shift = float(input("Enter the node alignment shift (from 0 to 0.5): ")) 
    print()

    if para == 0:
        selectedFiberLx = x - (shift * 0.5)
        selectedFiberLy = np.ones(len(selectedFiberLx)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLz = np.ones(len(selectedFiberLx)) * (6.75 + fiber_v_displacement)    #specifying nodal locations for a fiber
    else:
        selectedFiberLz = z - (shift * 0.5)
        selectedFiberLy = np.ones(len(selectedFiberLz)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLx = np.ones(len(selectedFiberLz)) * 0    #specifying nodal locations for a fiber

    # Get compartmental EC potentials from previously made 3d-grid
    for i in range(len(selectedFiberLx)):
        try:
                    
            fiberLVoltages[i]=(grid_e1( [selectedFiberLx[i], selectedFiberLy[i], selectedFiberLz[i]] ))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range!")
            pass
                
    # fiberLVoltages_nodes = np.multiply(fiberLVoltages, -1)
    # secSpatialDiffs = list()
    # for i in range(1,len(fiberLVoltages_nodes)-1):
    #     temp_ssd = (fiberLVoltages_nodes[i-1] - (2*fiberLVoltages_nodes[i]) + fiberLVoltages_nodes[i+1])
    #     secSpatialDiffs.append(temp_ssd)

    # figAllGood = plt.figure()
    # x_axis = range(len(fiberLVoltages_nodes))
    # ax = figAllGood.add_subplot()
    # ax.plot(x_axis,((np.array(fiberLVoltages_nodes))),'.-b', lw=1, ms=4)
    
    # ax.grid()
    # ax1 = ax.twinx()
    # ax1.plot(x_axis[1:len(x_axis)-1],secSpatialDiffs, '.-g', lw=1, ms=4, label='second spatial difference')
    # #ax1.grid()
    # plt.show()

    fiberLVoltages_nodes = fiberLVoltages
    voltages = [k*-1 for k in fiberLVoltages_nodes]

    ssds = []
    fsds = []

    for ind in range(1,len(voltages)-1):
        temp_ssd = voltages[ind-1] - (2*voltages[ind]) + voltages[ind+1]
        ssds.append(temp_ssd)

        temp_fsd = (voltages[ind+1] - voltages[ind-1]) / 2
        fsds.append(temp_fsd)

    min_volt_ind = 0
    min_volt = 0
    for i in range(len(voltages)):
        if voltages[i] < min_volt:
            min_volt = voltages[i]
            min_volt_ind = i

    fig = plt.figure(figsize=(15,5))
    x_axis = range(len(voltages))

    ax = fig.add_subplot(131)
    ax.plot(x_axis,((np.array(voltages))),'.-k', lw=1, ms=6)
    ax.plot(x_axis[min_volt_ind-5:min_volt_ind+6],((np.array(voltages[min_volt_ind-5:min_volt_ind+6]))),'.r', lw=1, ms=6)

    ax = fig.add_subplot(132)
    ax.plot(x_axis[1:len(x_axis)-1],((np.array(fsds))),'.-k', lw=1, ms=6)
    ax.plot(x_axis[min_volt_ind-5:min_volt_ind+6],((np.array(fsds[min_volt_ind-5-1:min_volt_ind+6-1]))),'.r', lw=1, ms=6)

    ax = fig.add_subplot(133)
    ax.plot(x_axis[1:len(x_axis)-1],((np.array(ssds))),'.-k', lw=1, ms=6)
    ax.plot(x_axis[min_volt_ind-5:min_volt_ind+6],((np.array(ssds[min_volt_ind-5-1:min_volt_ind+6-1]))),'.r', lw=1, ms=6)

    plt.show()