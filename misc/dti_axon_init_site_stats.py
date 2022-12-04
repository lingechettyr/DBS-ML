'''
    This script is used to analyze the result jsons of dti_axon_init_site_finder.py
    to compute statistics regarding the MRG predicted activation initiation site vs
    the extracellular nodal voltage and second-spatial-difference profiles.
'''

import json
import numpy as np
from pathlib import Path
import sys
import os
from matplotlib import pyplot as plt

input_path = sys.argv[1] # name of input json path

voltage_init_diff_arry = []
ssd_init_diff_arry = []
voltage_ssd_diff_arry = []

# Check all files in input path
for filename in Path(input_path).rglob("*.json"):
    json_filename = os.path.basename(filename)

    with open(filename) as f:
        preData = json.load(f)

    act_inds = preData["Activation_Properties"]["initiation_nodes"]
    voltages = preData["Extracellular_Potential_Properties"]["ec_potentials"]

    if len(act_inds) == 1:
        act_ind = act_inds[0]
    elif len(act_inds) == 2:
        act_ind = (act_inds[0] + act_inds[1]) / 2
        continue
    elif len(act_inds) == 3:
        act_ind = act_inds[1]
    else:
        print("***FLAG*** handle this if ever encountered")

    max_voltage = 20
    max_voltage_node = 0          
    max_ssd = -20
    max_ssd_node = 0
    for ind in range(1,len(voltages)-1):
        temp_ssd = voltages[ind-1] - (2*voltages[ind]) + voltages[ind+1]

        if temp_ssd > max_ssd:
            max_ssd = temp_ssd
            max_ssd_node = ind

        if voltages[ind] < max_voltage:
            max_voltage = voltages[ind]
            max_voltage_node = ind

    voltage_init_diff = abs(max_voltage_node - act_ind)
    ssd_init_diff = abs(max_ssd_node - act_ind)
    voltage_ssd_diff = abs(max_voltage_node - max_ssd_node)

    voltage_init_diff_arry.append(voltage_init_diff)
    ssd_init_diff_arry.append(ssd_init_diff)
    voltage_ssd_diff_arry.append(voltage_ssd_diff)
 

num_fibers = len(voltage_init_diff_arry)

print()
print("Stats for " + str(num_fibers) + " fiber simulations:")
print()
print("Average nodal distance between the max voltage and initiation site: " + str(np.average(voltage_init_diff_arry)))
print("Average nodal distance between the max ssd and initiation site: " + str(np.average(ssd_init_diff_arry)))
print("Average nodal distance between the max voltage and max ssd: " + str(np.average(voltage_ssd_diff_arry)))
print()
print("Median nodal distance between the max voltage and initiation site: " + str(np.median(voltage_init_diff_arry)))
print("Median nodal distance between the max ssd and initiation site: " + str(np.median(ssd_init_diff_arry)))
print("Median nodal distance between the max voltage and max ssd: " + str(np.median(voltage_ssd_diff_arry)))
print()

upper_hist_bound = 21
hist_bins = list(range(0,upper_hist_bound)) #[0,1,2,3,4,5,6,7,8,9,10]
#print(hist_bins)
hist_bins.append(300)
x_limit = upper_hist_bound
y_limit = 150

fig = plt.figure(figsize=(21,7))
ax = fig.add_subplot(131)
ax.hist(voltage_init_diff_arry, bins=hist_bins)
title_str = "Voltage Peak to Initiation Site"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_xticks(hist_bins[:len(hist_bins)-1])
ax.set_ylim(0,y_limit)
ax.grid()

ax = fig.add_subplot(132)
ax.hist(ssd_init_diff_arry, bins=hist_bins)
title_str = "SSD Peak to Initiation Site"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_xticks(hist_bins[:len(hist_bins)-1])
ax.set_ylim(0,y_limit)
ax.grid()

ax = fig.add_subplot(133)
ax.hist(voltage_ssd_diff_arry, bins=hist_bins)
title_str = "Voltage Peak to SSD Peak"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_xticks(hist_bins[:len(hist_bins)-1])
ax.set_ylim(0,y_limit)
ax.grid()

plt.show()
