'''
    This script predicts the activation of the straight fiber trajectories
    using an ANN, across all desired pulse widths. The orientation of the
    fibers (orthogonal or parallel) to the DBS lead is specified by the 
    user. The results are written out to a json LUT.
'''

# Imports
import sys
import numpy as np
import json
import math
import time

sys.path.append("../")
from lib.COMSOL import FEM
from ANN import ann_predict_lib


# COMMAND LINE INPUTS:
electrode1File = sys.argv[1] 
ANN_model = sys.argv[2]
output_json = sys.argv[3]
para = int(sys.argv[4])

SSD_centered = 0

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
node_to_node = 0.5
number_of_nodes = 41

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()
fem_bounds = fem.getFEMBounds()

ANN_model = ann_predict_lib.ANN(ANN_model)
ann_hparam_dict = ANN_model.get_hparam_dict()
num_ecs = ann_hparam_dict["num_ecs"]
num_fsds = ann_hparam_dict["num_fsds"]
num_ssds = ann_hparam_dict["num_ssds"]
ANN_REGRESSION = ann_hparam_dict["regression"]

input_array = []
problem_inds = []

h_distances = np.arange(0.5, 12.5+0.8, 0.8)
v_displacements = np.arange(-12, 12+0.8, 0.8)

h_distances = [round(k,1) for k in h_distances]
v_displacements = [round(k,1) for k in v_displacements]

orthogonal_offset = 0 # center of the active contact in the X dimension
parallel_offset = 6.75 # center of the active contact in the Z dimension

if para == 0:
    used_offset = orthogonal_offset
else:
    used_offset = parallel_offset

node_array = np.arange(0, (number_of_nodes*node_to_node), node_to_node)
pos_arry = node_array + used_offset - (((number_of_nodes-1)/2)*node_to_node)

for v_disp in v_displacements:
    for h_dist in h_distances:
    # Get compartmental EC potentials from previously made 3d-grid
        fiberLVoltages = []
        for i in range(len(pos_arry)):

            if para == 0:
                selectedFiberLx = pos_arry
                selectedFiberLy = np.ones(len(selectedFiberLx)) * (0.635 + h_dist) # the electrode is along the z-axis at x=0, y=0.4
                selectedFiberLz = np.ones(len(selectedFiberLx)) * (6.75 + v_disp)    #specifying nodal locations for a fiber
            else:
                selectedFiberLz = pos_arry
                selectedFiberLy = np.ones(len(selectedFiberLz)) * (0.635 + h_dist) # the electrode is along the z-axis at x=0, y=0.4
                selectedFiberLx = np.ones(len(selectedFiberLz)) * 0    #specifying nodal locations for a fiber
                
            try:
                        
                fiberLVoltages.append(float(grid_e1( [selectedFiberLx[i], selectedFiberLy[i], selectedFiberLz[i]] )))
                        
            except Exception as e:
                print("WARNING: 3d-position out of COMSOL range! X = " + str(xNodeComp[fib][i]) + ", Y = " + str(yNodeComp[fib][i]) + ", Z = " + str(zNodeComp[fib][i]))
                pass
                    
        fiberLVoltages_nodes = fiberLVoltages

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

    if para == 1:
        break

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
result_dict["problem_inds"] = problem_inds

if para == 1:
    v_displacements = [0.0]

itr = 0
for v_disp in v_displacements:
    for h_dist in h_distances:
        for pulse_width in pulse_widths:
            pwd = pulse_width / 1000

            if pwd not in result_dict:
                result_dict[pwd] = {}
            if h_dist not in result_dict[pwd]:
                result_dict[pwd][h_dist] = {}  

            result_dict[pwd][h_dist][v_disp] = float(ANN_prediction[itr])
            itr += 1

with open(output_json, 'w') as outfile:
    json.dump(result_dict, outfile, indent=4)


######################################################################################################