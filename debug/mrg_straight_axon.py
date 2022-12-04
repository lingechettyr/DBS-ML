'''
    This script allows the user to simulate individual fibers of specified orientation, 
    position, pulse width, amplitude, and node alignment, using the MRG axon model.
'''

# Imports
import sys
import numpy as np
import time
from neuron import h, gui
import json
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

sys.path.append("../")
from lib.NEURON import axon
from lib.COMSOL import FEM

h.nrn_load_dll("../lib/NEURON/nrnmech.dll")

# COMMAND LINE INPUTS: 
electrode1File = sys.argv[1] 
fiber_h_distance = float(sys.argv[2]) # the horizontal between the electrode lead and the fiber, in mm
fiber_v_displacement = float(sys.argv[3]) # the vertical displacement of the fiber center from the lead plane, in mm
number_of_nodes = int(sys.argv[4])
para = int(sys.argv[5])

showGraph = 0
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
    multiplier = float(input("Enter the EC multiplier: ")) 

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

    spikes, AP_init_inds = axon.stimulate(voltages1, multiplier, pwd, frequency, num_wfs)
    print(spikes)
    print(AP_init_inds)

    # threshold = axon.findThreshold(voltages1, pwd, frequency, num_wfs)
    # print(threshold)

    fiberLVoltages_nodes = fiberLVoltages[::15]

    ## This is a PoC Test, will need to change code structure to actually implement
    # x_int = selectedFiberLx[::15]
    # y_int = fiberLVoltages_nodes
    # cs = CubicSpline(x_int,y_int)
    # voltages_new = cs(selectedFiberLx)
    # threshold = axon.findThreshold(voltages_new, pwd, frequency, num_wfs)
    # print(threshold)

    x_axis = selectedFiberLx
    x_axis_nodes = selectedFiberLx[::15]
    secSpatialDiffs = list()
    for i in range(1,len(fiberLVoltages_nodes)-1):
        temp_ssd = (fiberLVoltages_nodes[i-1] - (2*fiberLVoltages_nodes[i]) + fiberLVoltages_nodes[i+1])
        secSpatialDiffs.append(temp_ssd)

    if showGraph == 1:
        figAllGood = plt.figure()
        
        ax = figAllGood.add_subplot()
        ax.plot(x_axis,((np.array(fiberLVoltages))),'.b', lw=1, ms=1)
        ax.plot(x_axis_nodes,((np.array(fiberLVoltages_nodes))),'.r', lw=1, ms=4)
        ax.plot()
        
        ax.grid()
        ax1 = ax.twinx()
        ax1.plot(x_axis_nodes[1:len(x_axis_nodes)-1],secSpatialDiffs, '.-g', lw=1, ms=4, label='second spatial difference')
        plt.show()
             
######################################################################################################