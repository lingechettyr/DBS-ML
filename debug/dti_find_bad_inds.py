'''
    This script lets the user find the DTI-fiber indices of the fibers
    that resulted in errors above a specified threshold.
'''

from matplotlib import pyplot as plt
import numpy as np
import sys
import json
import pandas as pd

NEURON_data_json = sys.argv[1]
model_data_json = sys.argv[2]
pulse_width = float(sys.argv[3]) / 1000

error_bound = 20

# Open NEURON result dictionary
with open(NEURON_data_json) as f:
    NEURON_data = json.load(f)

# Open comparison model result dictionary
with open(model_data_json) as f:
    model_data = json.load(f)

num_total_fibers = len(model_data[str(pulse_width)])
model_problem_inds = model_data["problem_inds"]

NEURON_ths = []
model_ths = []

for fib_ind in range(num_total_fibers):
    NEURON_ths.append(NEURON_data[str(pulse_width)][str(fib_ind)])
    model_ths.append(model_data[str(pulse_width)][str(fib_ind)])

model_abs_error = np.subtract(model_ths, NEURON_ths)
model_perc_error = np.multiply(np.divide(model_abs_error, NEURON_ths), 100)

## Find indices of bad fibers
bad_inds = []
for ind in range(len(NEURON_ths)):
    if model_ths[ind] - NEURON_ths[ind] > error_bound and ind not in model_problem_inds:
        bad_inds.append(ind)

print(bad_inds)
