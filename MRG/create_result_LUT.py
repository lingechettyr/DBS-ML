'''
    This script is used to parse all json output files from HPG,
    formatting the data as entries into a json look-up-table (LUT).
    The user can select whether jsons from straight fiber sims or 
    DTI-tractography sims are being processed with the third input
    argument.
'''

import json
import numpy as np
from pathlib import Path
import sys
import os

input_path = sys.argv[1] # name of input json path
output_json = sys.argv[2] # name of output json
for_DTI = int(sys.argv[3]) # 0 if processing straight fiber jsons, 1 if processing DTI fiber jsons

FC_dict = {}

# Check all files in input path, can this be improved somehow?
for filename in Path(input_path).rglob("*.json"):
    json_filename = os.path.basename(filename)
    try:
        with open(filename) as f:
            preData = json.load(f)
    except:
        ## This means there is an empty json, usually because the MRG HPG job got stopped abruptly ##
        print(filename)
        continue

    # Get relevant data from the jsons
    pulse_width = preData["Stimulus_Properties"]["pulse_width"]
    threshold = preData['Activation_Properties']['threshold_multiplier']

    if pulse_width not in FC_dict:
        FC_dict[pulse_width] = {}

    if for_DTI == 0:
        h_distance = preData["Fiber_Properties"]["horizontal_distance"]
        v_displacement = preData["Fiber_Properties"]["vertical_displacement"]

        if h_distance not in FC_dict[pulse_width]:
            FC_dict[pulse_width][h_distance] = {}  

        FC_dict[pulse_width][h_distance][v_displacement] = threshold

    else:
        splits = json_filename.split('_')
        dti_index = splits[1]

        FC_dict[pulse_width][dti_index] = threshold       

with open(output_json, 'w') as outfile:
    json.dump(FC_dict, outfile, indent=4)