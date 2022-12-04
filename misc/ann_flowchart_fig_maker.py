# Imports
import sys
import numpy as np
import json
import math
import time
import matplotlib.pyplot as plt

sys.path.append("../")
from lib.DTI import process_DTI
from lib.DTI import graph_DTI
from lib.COMSOL import FEM


# COMMAND LINE INPUTS:
electrode1File = sys.argv[1]
tract_filename = sys.argv[2]

node_to_node = 0.5

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
fem_bounds = fem.getFEMBounds()

test_dti = process_DTI.DTI_tracts(tract_filename, fem_bounds, node_to_node)
xNodeComp, yNodeComp, zNodeComp = test_dti.getNodeCompPos()
test_dti_graph = graph_DTI.DTI_grapher(test_dti.getLeadCoordinates(), xNodeComp, yNodeComp, zNodeComp)

print()
print("Number of fibers = " + str(len(xNodeComp)))
print()

while(True):
    fib = int(input("Enter the index of a fiber within range: "))
    # Get compartmental EC potentials from previously made 3d-grid
    fiberLVoltages = []
    for i in range(len(xNodeComp[fib])):
        try:
                    
            fiberLVoltages.append(grid_e1( [xNodeComp[fib][i], yNodeComp[fib][i], zNodeComp[fib][i]] ))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xNodeComp[fib][i]) + ", Y = " + str(yNodeComp[fib][i]) + ", Z = " + str(zNodeComp[fib][i]))
            pass
                
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

    # fig = plt.figure(figsize=(15,5))
    # x_axis = range(len(voltages))

    # ax = fig.add_subplot(131)
    # ax.plot(x_axis,((np.array(voltages))),'.-k', lw=1, ms=6)

    # ax = fig.add_subplot(132)
    # ax.plot(x_axis[1:len(x_axis)-1],((np.array(fsds))),'.-k', lw=1, ms=6)

    # ax = fig.add_subplot(133)
    # ax.plot(x_axis[1:len(x_axis)-1],((np.array(ssds))),'.-k', lw=1, ms=6)

    plt.show()

    test_dti_graph.plotSingleTractColorMap(fib, fiberLVoltages_nodes)


      
######################################################################################################