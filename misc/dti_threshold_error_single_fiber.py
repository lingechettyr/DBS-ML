'''
    This script plots error scatterplots for the predicted thresholds of the MRG model
    and a selected predictor model. The predictions must be passed via json LUTs.
'''

from matplotlib import pyplot as plt
import numpy as np
import sys
import json
import pandas as pd

PLT_THRESHOLDS = 1
PLT_PERCENT_ERROR = 0
PLT_ABS_ERROR = 0

#pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
pulse_widths = [60,  90,  120,  150,  200,  250,  300,  400,  500]
#pulse_widths = [ 75,  105,  135,  175,  225,  275,  350,  450]

NEURON_data_json = sys.argv[1]
model_data_json = sys.argv[2]

# Open NEURON result dictionary
with open(NEURON_data_json) as f:
    NEURON_data = json.load(f)

# Open comparison model result dictionary
with open(model_data_json) as f:
    model_data = json.load(f)

model_problem_inds = model_data["problem_inds"]
print(model_problem_inds)
print()

while(True):
    fiber_ind = int(input("Enter the fiber index: "))

    NEURON_ths = []
    model_ths = []

    if fiber_ind in model_problem_inds:
        print("The fiber is a problem fiber, choose a different index...")
    else:
        for pulse_width in pulse_widths:
            NEURON_ths.append(NEURON_data[str(pulse_width/1000)][str(fiber_ind)])
            model_ths.append(model_data[str(pulse_width/1000)][str(fiber_ind)])

            model_abs_error = np.subtract(model_ths, NEURON_ths)
            model_perc_error = np.multiply(np.divide(model_abs_error, NEURON_ths), 100)

        if PLT_THRESHOLDS == 1:
            fig = plt.figure(figsize=(7,7))
            ax = plt.subplot()
            ax.scatter(pulse_widths, NEURON_ths, s=24, c="black", label="MRG")
            ax.scatter(pulse_widths, model_ths, s=24, c="red", label="ANN")
            #ax.set(xlabel='Pulse Width (us)', ylabel='Threshold (V)', title='MRG vs ANN Thresholds')
            #plt.xlim(0,10)
            #plt.ylim(-150,150)
            plt.xlabel('Pulse Width (us)', fontsize=20)
            plt.ylabel('Threshold (V)',fontsize=20)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            plt.legend(fontsize=20)
            plt.show()

        if PLT_PERCENT_ERROR == 1:
            fig = plt.figure(figsize=(7,7))
            ax = plt.subplot()
            ax.scatter(pulse_widths, model_perc_error,s=3, c="black")
            ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (%)', title='ANN Threshold Error (%)')
            #plt.xlim(0,10)
            #plt.ylim(-150,150)
            plt.show()

        if PLT_ABS_ERROR == 1:
            fig = plt.figure(figsize=(7,7))
            ax = plt.subplot()
            ax.scatter(pulse_widths, model_abs_error,s=3, c="black")
            ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (V)', title='ANN Threshold Error (V)')
            #plt.xlim(0,10)
            #plt.ylim(-150,150)
            plt.show()
