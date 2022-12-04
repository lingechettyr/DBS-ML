'''
    This script predicts the activation of the DTI-tractography based fiber trajectories
    using an ANN. The results are graphed using the DTI_graphing library with Mayavi.
'''

# Imports
import sys
import numpy as np
import json
import math
import time

sys.path.append("../")
from lib.DTI import process_DTI
from lib.DTI import graph_DTI
from lib.COMSOL import FEM
from ANN import ann_predict_lib


# COMMAND LINE INPUTS:
electrode1File = sys.argv[1]
tractFile = sys.argv[2]
ANN_model = sys.argv[3]

SSD_centered = 0

node_to_node = 0.5

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
fem_bounds = fem.getFEMBounds()

test_dti = process_DTI.DTI_tracts(tractFile, fem_bounds, node_to_node)

ANN_model = ann_predict_lib.ANN(ANN_model)
ann_hparam_dict = ANN_model.get_hparam_dict()
num_ecs = ann_hparam_dict["num_ecs"]
num_fsds = ann_hparam_dict["num_fsds"]
num_ssds = ann_hparam_dict["num_ssds"]
ANN_REGRESSION = ann_hparam_dict["regression"]

xNodeComp, yNodeComp, zNodeComp = test_dti.getNodeCompPos()
preTruncxNodeComp, preTruncyNodeComp, preTrunczNodeComp = test_dti.getPreTruncNodeCompPos()
# test_dti_graph = graph_DTI.DTI_grapher(test_dti.getLeadCoordinates(), preTruncxNodeComp, preTruncyNodeComp, preTrunczNodeComp)
test_dti_graph = graph_DTI.DTI_grapher(test_dti.getLeadCoordinates(), xNodeComp, yNodeComp, zNodeComp)

input_array = []
problem_inds = []

voltages_matrix = []
max_voltage = 0

for fib in range(len(xNodeComp)):
    # Get compartmental EC potentials from previously made 3d-grid
    fiberLVoltages = []
    for i in range(len(xNodeComp[fib])):
        try:
                    
            fiberLVoltages.append(grid_e1( [xNodeComp[fib][i], yNodeComp[fib][i], zNodeComp[fib][i]] ))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xNodeComp[fib][i]) + ", Y = " + str(yNodeComp[fib][i]) + ", Z = " + str(zNodeComp[fib][i]))
            pass
                
    fiberLVoltages_nodes = fiberLVoltages
    voltages_matrix.append(fiberLVoltages_nodes)
    temp_max = max(fiberLVoltages_nodes)
    if temp_max > max_voltage:
        max_voltage = temp_max

    ssds = []
    ecs = []
    fsds = []

    max_ssd_ind = 0
    max_ssd = 0
    for ind in range(1,len(fiberLVoltages_nodes)-1):
        temp_ssd = -1 * (fiberLVoltages_nodes[ind-1] - (2*fiberLVoltages_nodes[ind]) + fiberLVoltages_nodes[ind+1])
        if temp_ssd > max_ssd:
            max_ssd = temp_ssd
            max_ssd_ind = ind

    max_ec_ind = 0
    max_ec = 0
    for ind in range(len(fiberLVoltages_nodes)):
        temp_ec = fiberLVoltages_nodes[ind]
        if temp_ec > max_ec:
            max_ec = temp_ec
            max_ec_ind = ind

    center_ind = 0
    if SSD_centered == 1:
        center_ind = max_ssd_ind
    else:
        center_ind = max_ec_ind

    input_sizes = [num_ecs, num_fsds, num_ssds]
    largest_size = max(input_sizes)
    input_bound = math.floor(largest_size / 2) + 1

    if center_ind < input_bound or len(fiberLVoltages_nodes) - 1 - center_ind < input_bound or max_ec_ind < 2 or len(fiberLVoltages_nodes) - 1 - max_ec_ind < 2:
        problem_inds.append(fib)
        ssds = []
        ecs = []
        fsds = []
        for ind in range(center_ind - (math.floor(num_ecs / 2)), center_ind + (math.floor(num_ecs / 2)) + 1):
            ecs.append(0)

        for ind in range(center_ind - (math.floor(num_fsds / 2)), center_ind + (math.floor(num_fsds / 2)) + 1):
            fsds.append(0)

        for ind in range(center_ind - (math.floor(num_ssds / 2)), center_ind + (math.floor(num_ssds / 2)) + 1):
            ssds.append(0)

        if num_ecs == 0:
            ecs.pop()
        if num_fsds == 0:
            fsds.pop()
        if num_ssds == 0:
            ssds.pop()
    else:
        for ind in range(center_ind - (math.floor(num_ecs / 2)), center_ind + (math.floor(num_ecs / 2)) + 1):
            temp_ec = fiberLVoltages_nodes[ind] # stimulus is cathodic, so multiply by -1
            ecs.append(temp_ec)

        for ind in range(center_ind - (math.floor(num_fsds / 2)), center_ind + (math.floor(num_fsds / 2)) + 1):
            temp_fsd = -1 * (fiberLVoltages_nodes[ind+1] - fiberLVoltages_nodes[ind-1]) / 2 
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

    test_features = []
    test_features.extend(ecs)
    test_features.extend(fsds)
    test_features.extend(ssds)

    input_array.append(test_features)

print(problem_inds)
print()

test_dti_graph.plotTractsColorMap(voltages_matrix, max_voltage)

while(True):

    pwd = float(input("Enter the stimulus pulse width in us: ")) / 1000
    multiplier = float(input("Enter the stimulus multiplier: "))
    fiber_ind = int(input("Enter fiber index within range: "))

    if ANN_REGRESSION == 0:
        prediction_start_time = time.time()
        test_features = []
        for i in range(len(input_array)):
            tensor = np.multiply(np.asarray(input_array[i]), multiplier)
            tensor = np.insert(tensor, 0, pwd)
            test_features.append(tensor)

        ANN_prediction = ANN_model.batch_predict(test_features)

        act_inds = []
        for i in range(len(ANN_prediction)):
            if ANN_prediction[i] >= 0.5:
                act_inds.append(i)
        
        prediction_stop_time = time.time()

    elif ANN_REGRESSION == 1:
        prediction_start_time = time.time()
        test_features = []
        for i in range(len(input_array)):
            tensor = np.asarray(input_array[i])
            tensor = np.insert(tensor, 0, pwd)
            test_features.append(tensor)

        ANN_prediction = ANN_model.batch_predict_threshold_reg(test_features)

        act_inds = []
        for i in range(len(ANN_prediction)):
            if ANN_prediction[i] <= multiplier:
                act_inds.append(i)
        
        prediction_stop_time = time.time()

    print()
    print(str(len(test_features)) + " ANN predictions took " + str(prediction_stop_time - prediction_start_time) + " s")
    print(str(len(act_inds)) + " fibers are predicted to activate")
    print()

    #test_dti_graph.plotActivatedTracts(act_inds, problem_inds)
    print(act_inds)
    test_dti_graph.plotActivatedTracts_temp(act_inds, problem_inds, fiber_ind)
    test_dti_graph.plotTractsColorMap_temp(voltages_matrix, max_voltage, fiber_ind)


      
######################################################################################################