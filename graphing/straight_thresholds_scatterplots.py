'''
    This script creates scatterplots of the error between a selected predictor model
    and NEURON results. Each model's results must be passed via json LUTs.
'''

# Imports
import sys
import numpy as np
from matplotlib import pyplot as plt
import json
import pandas as pd

NEURON_data_json = sys.argv[1]
model_data_json = sys.argv[2]
para = int(sys.argv[3])

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]

PLT_ABS_ERROR_SCATTERPLT = 1
PLT_PERCENT_ERROR_SCATTERPLT = 1
PLT_THRESHOLD_SCATTERPLT = 1

# Open NEURON result dictionary
with open(NEURON_data_json) as f:
    NEURON_data = json.load(f)

# Open comparison model result dictionary
with open(model_data_json) as f:
    model_data = json.load(f)

h_distances = np.arange(0.5, 6.5+0.2, 0.2)
v_displacements = np.arange(-4.8, 4.8+0.2, 0.2)

h_distances = [round(k,1) for k in h_distances]
v_displacements = [round(k,1) for k in v_displacements]

if para == 1:
    v_displacements = [0.0]

NEURON_ths = np.empty((len(pulse_widths), len(h_distances) * len(v_displacements)))
model_ths = np.empty((len(pulse_widths), len(h_distances) * len(v_displacements)))

for pwd_ind in range(len(pulse_widths)):
    itr = 0
    for h_dist in h_distances:
        for v_disp in v_displacements:
            NEURON_ths[pwd_ind][itr] = NEURON_data[str(pulse_widths[pwd_ind]/1000)][str(h_dist)][str(v_disp)]
            model_ths[pwd_ind][itr] = model_data[str(pulse_widths[pwd_ind]/1000)][str(h_dist)][str(v_disp)]
            itr += 1

model_abs_error = np.subtract(model_ths, NEURON_ths)
model_perc_error = np.multiply(np.divide(model_abs_error, NEURON_ths), 100)
print("Model average |absolute| error (V) = " + str(np.average(np.absolute(model_abs_error))))
print("Model average |percent| error = " + str(np.average(np.absolute(model_perc_error))))
#print(len(np.where(np.absolute(ANN_perc_error) < 1)[0]) / (ANN_abs_error.shape[0] * ANN_abs_error.shape[1]))

print()
print("Number of pulse widths: " + str(model_abs_error.shape[0]))
print("Number of data points per pulse width: " + str(model_abs_error.shape[1]))

NEURON_ths_flat = np.reshape(NEURON_ths, (NEURON_ths.shape[0]*NEURON_ths.shape[1]))
model_ths_flat = np.reshape(model_ths, (model_ths.shape[0]*model_ths.shape[1]))
model_abs_error_flat = np.reshape(model_abs_error, (model_abs_error.shape[0]*model_ths.shape[1]))
model_perc_error_flat = np.reshape(model_perc_error, (model_perc_error.shape[0]*model_ths.shape[1]))

if PLT_ABS_ERROR_SCATTERPLT == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.scatter(NEURON_ths_flat, model_abs_error_flat,s=3, c="black")
    ax.set(xlabel='Amplitude (V)', ylabel='Threshold Error (V)', title='Model Predicted Threshold Error')
    #plt.xlim(0,10)
    plt.show()

if PLT_PERCENT_ERROR_SCATTERPLT == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.scatter(NEURON_ths_flat, model_perc_error_flat,s=3, c="black")
    ax.set(xlabel='Amplitude (V)', ylabel='Threshold Error (%)', title='Model Predicted Threshold Error (%)')
    #plt.xlim(0,10)
    plt.show()

if PLT_THRESHOLD_SCATTERPLT == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.scatter(NEURON_ths_flat, model_ths_flat,s=3, c="black")
    ax.set(xlabel='NEURON Vths', ylabel='Model Vths', title='Model Predicted Threshold Error')
    plt.xlim(0,10)
    plt.ylim(0,10)
    plt.show()