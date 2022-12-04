'''
    This script predicts the activation of the DTI-tractography based fiber trajectories
    using an ANN, across all desired pulse widths. The results are written out to a json
    LUT, to be compared with result LUTs from other predictor models.
'''

# Imports
import sys
import numpy as np
import json
import math
import time

sys.path.append("../")
from lib.DTI import process_DTI
from lib.COMSOL import FEM
from ANN import ann_predict_lib


# COMMAND LINE INPUTS: 
electrode1File = sys.argv[1]
tractFile = sys.argv[2]
ANN_model = sys.argv[3]
output_json = sys.argv[4]

SSD_CENTERED = 0

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
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

input_array = []
problem_inds = []

for fib in range(len(xNodeComp)):
    # Get compartmental EC potentials from previously made 3d-grid
    fiberLVoltages = []
    for i in range(len(xNodeComp[fib])):
        try:
                    
            fiberLVoltages.append(float(grid_e1( [xNodeComp[fib][i], yNodeComp[fib][i], zNodeComp[fib][i]] )))
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range! X = " + str(xNodeComp[fib][i]) + ", Y = " + str(yNodeComp[fib][i]) + ", Z = " + str(zNodeComp[fib][i]))
            pass
                
    fiberLVoltages_nodes = fiberLVoltages

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
    if SSD_CENTERED == 1:
        center_ind = max_ssd_ind
    else:
        center_ind = max_ec_ind

    input_sizes = [num_ecs, num_fsds, num_ssds]
    largest_size = max(input_sizes)
    input_bound = math.floor(largest_size / 2) + 1

    if center_ind < input_bound or len(fiberLVoltages_nodes) - 1 - center_ind < input_bound or max_ec_ind < 2 or len(fiberLVoltages_nodes) - 1 - max_ec_ind < 2:
        problem_inds.append(fib)

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
      
    for pwd in pulse_widths:
    
        test_features = [pwd/1000]
        test_features.extend(ecs)
        test_features.extend(fsds)
        test_features.extend(ssds)

        input_array.append(test_features)

if ANN_REGRESSION == 1:
    prediction_start_time = time.time()
    ANN_prediction = ANN_model.batch_predict_threshold_reg(input_array)
    prediction_stop_time = time.time()
else:
    prediction_start_time = time.time()
    ANN_prediction = ANN_model.batch_predict_threshold(input_array, 100)
    prediction_stop_time = time.time()

print()
print(str(len(input_array)) + " ANN predictions took " + str(prediction_stop_time - prediction_start_time) + " s")
print()

result_dict = {}

total_fibs = len(xNodeComp) 
good_fibs = total_fibs - len(problem_inds)
pulse_width_count = len(pulse_widths)

result_dict["problem_inds"] = problem_inds

for i in range(len(ANN_prediction)):
    pulse_width = pulse_widths[int(i % pulse_width_count)] / 1000
    dti_index = int(i / pulse_width_count)
    
    if pulse_width not in result_dict:
        result_dict[pulse_width] = {}

    result_dict[pulse_width][dti_index] = float(ANN_prediction[i]) 

with open(output_json, 'w') as outfile:
    json.dump(result_dict, outfile, indent=4)

######################################################################################################