'''
    This script allows the user to create a strength duration curve by simulating
    the MRG compartment model. The stimulus pulse width is swept across a desired 
    range, and the thresholds plotted as a function of pulse width.
'''

# Imports
import sys
import numpy as np
import time
from neuron import h, gui
import json
import matplotlib.pyplot as plt

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
shift = float(sys.argv[6])

showGraph = 1
diameter = 5.7
STINmul = 10
compartDiv = STINmul + 5
node_to_node = .5
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

pwds = np.arange(15, 500+1, 30)
thresholds = []

for pwd in pwds:

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
            print("WARNING: 3d-position out of COMSOL range! X = " + str(selectedFiberLx[i]) + ", Y = " + str(selectedFiberLy[i]) + ", Z = " + str(selectedFiberLz[i]))
            pass
                
    voltages1 = fiberLVoltages

    threshold = axon.findThreshold(voltages1, pwd/1000, frequency, num_wfs)
    print(threshold)
    thresholds.append(threshold)

figAllGood = plt.figure()
ax = figAllGood.add_subplot()
ax.plot(pwds,(thresholds),'.-b', lw=1, ms=8)
ax.grid()
plt.show()
            
######################################################################################################