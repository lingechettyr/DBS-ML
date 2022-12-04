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

input_json = sys.argv[1] # name of input json
pulse_widths = [60, 500]
#pulse_widths_all = [60, 90, 120, 150, 200, 250, 300, 400, 500]
pulse_widths_all = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]

with open(input_json) as f:
    preData = json.load(f)

h_dists = np.arange(500,6500+1,200)

thresholds_60 = [] #np.empty(num_tracts)
for h_dist in h_dists:
    thresholds_60.append(preData[str(.06)][str(h_dist/1000)][str(0.0)])
    #thresholds[i] = preData[str(pulse_width)][str(i)]

thresholds_500 = [] #np.empty(num_tracts)
for h_dist in h_dists:
    thresholds_500.append(preData[str(.5)][str(h_dist/1000)][str(0.0)])

thresholds_all = []
for pwd in pulse_widths_all:
    for h_dist in h_dists:
        thresholds_all.append(preData[str(pwd/1000)][str(h_dist/1000)][str(0.0)])



# print("Number of total thresholds: " + str(len(thresholds)))
# thresholds_5 = [k for k in thresholds if k <= 5]#np.where(thresholds <= 10)
# print("Number of thresholds <= 5 V: " + str(len(thresholds_5)))
# thresholds_10 = [k for k in thresholds if k <= 10]#np.where(thresholds <= 10)
# print("Number of thresholds <= 10 V: " + str(len(thresholds_10)))
# thresholds_20 = [k for k in thresholds if k <= 20]#np.where(thresholds <= 10)
# print("Number of thresholds <= 20 V: " + str(len(thresholds_20)))
# thresholds_50 = [k for k in thresholds if k <= 50]#np.where(thresholds <= 10)
# print("Number of thresholds <= 50 V: " + str(len(thresholds_50)))

upper_hist_bound = 51
hist_bins = list(range(0,upper_hist_bound)) #[0,1,2,3,4,5,6,7,8,9,10]
#print(hist_bins)
#hist_bins.append(300)
x_limit = upper_hist_bound
y_limit = 20

fig = plt.figure(figsize=(14,7))
ax = fig.add_subplot(121)
ax.hist(thresholds_60, bins=hist_bins)
title_str = "60 us pulse width"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_ylim(0,y_limit)
ax.grid()

ax = fig.add_subplot(122)
ax.hist(thresholds_500, bins=hist_bins)
title_str = "500 us pulse width"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_ylim(0,y_limit)
ax.grid()

plt.show()

y_limit = 100
fig = plt.figure(figsize=(7,7))
ax = fig.add_subplot()
ax.hist(thresholds_all, bins=hist_bins)
title_str = "All pulse widths"
ax.set(title=title_str)
ax.set_xlim(0,x_limit)
ax.set_ylim(0,y_limit)
ax.grid()
plt.show()