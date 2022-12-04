'''
    This script converts a LUT json of straight fiber results into the same indexed format used 
    for DTI fiber results. This will let me use the same graphing scripts without having to 
    modify them.
'''

import numpy as np
import sys
import json

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
#pulse_widths = [60,  90,  120,  150,  200,  250,  300,  400,  500]
#pulse_widths = [ 75,  105,  135,  175,  225,  275,  350,  450]

h_distances = np.arange(0.5, 12.5+0.8, 0.8)
v_displacements = np.arange(-12, 12+0.8, 0.8)

h_distances = [round(k,1) for k in h_distances]
v_displacements = [round(k,1) for k in v_displacements]

data_json = sys.argv[1]
output_json = sys.argv[2]

# Open LUT json
with open(data_json) as f:
    data = json.load(f)

result_dict = {}

try:
    problem_inds = data["problem_inds"]
    result_dict["problem_inds"] = problem_inds
except:
    print("No problem inds field")

for pulse_width in pulse_widths:
    pwd = pulse_width / 1000
    result_dict[str(pwd)] = {}
    fiber_ind = 0
    for h_dist in h_distances:
        for v_disp in v_displacements:
            result_dict[str(pwd)][str(fiber_ind)] = data[str(pwd)][str(h_dist)][str(v_disp)]
            fiber_ind += 1
        
with open(output_json, 'w') as outfile:
    json.dump(result_dict, outfile, indent=4)