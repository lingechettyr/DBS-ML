'''
    This script plots error scatterplots for the predicted thresholds of the MRG model
    and a selected predictor model. The predictions must be passed via json LUTs.
'''

from matplotlib import pyplot as plt
import numpy as np
import sys
import json
import pandas as pd

pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
#pulse_widths = [60,  90,  120,  150,  200,  250,  300,  400,  500]
#pulse_widths = [ 75,  105,  135,  175,  225,  275,  350,  450]

upper_amp_bound = 10

PLT_SCATTERPLT_ABS_ERROR = 0
PLT_SCATTERPLT_PERCENT_ERROR = 0
PLT_THRESHOLD_SCATTERPLT = 0
PLT_THRESHOLD_HISTOGRAMS = 1

NEURON_ths = []
model_ths = []
cmd_line_itr = 1
exitCond = False

points_per_pathway = []

while(exitCond == False):
    try:
        NEURON_data_json = sys.argv[cmd_line_itr]
        model_data_json = sys.argv[cmd_line_itr+1]

        # Open NEURON result dictionary
        with open(NEURON_data_json) as f:
            NEURON_data = json.load(f)

        # Open comparison model result dictionary
        with open(model_data_json) as f:
            model_data = json.load(f)

        model_problem_inds = model_data["problem_inds"]

        points_cnt = 0

        for pulse_width in pulse_widths:
            for fib_ind in range(len(model_data[str(pulse_width/1000)])):
                if fib_ind not in model_problem_inds and (NEURON_data[str(pulse_width/1000)][str(fib_ind)] <= upper_amp_bound or model_data[str(pulse_width/1000)][str(fib_ind)] <= upper_amp_bound):
                    NEURON_ths.append(NEURON_data[str(pulse_width/1000)][str(fib_ind)])
                    model_ths.append(model_data[str(pulse_width/1000)][str(fib_ind)])
                    points_cnt += 1

        points_per_pathway.append(points_cnt)
        cmd_line_itr += 2
    except:
        exitCond = True


model_abs_error = np.subtract(model_ths, NEURON_ths)
model_perc_error = np.multiply(np.divide(model_abs_error, NEURON_ths), 100)

print("Number of comparisons = " + str(len(NEURON_ths)))
print("Average absolute error = " + str(np.average(np.abs(model_abs_error))))
print("Average absolute percent error = " + str(np.average(np.abs(model_perc_error))))

if PLT_SCATTERPLT_ABS_ERROR == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.scatter(NEURON_ths, model_abs_error,s=3, c="black")
    ax.plot([0,upper_amp_bound], [0,0],lw=1, c="red")
    ax.set(xlabel='Amplitude (V)', ylabel='Threshold Error (V)', title='Model Predicted Threshold Error')
    plt.xlim(0,10)
    #plt.ylim(-150,150)
    plt.show()

if PLT_SCATTERPLT_PERCENT_ERROR == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    ax.scatter(NEURON_ths, model_perc_error,s=3, c="black")
    ax.plot([0,upper_amp_bound], [0,0],lw=1, c="red")
    ax.set(xlabel='Amplitude (V)', ylabel='Threshold Error (%)', title='Model Predicted Threshold Error (%)')
    plt.xlim(0,10)
    #plt.ylim(-150,150)
    plt.show()

if PLT_THRESHOLD_SCATTERPLT == 1:
    upper_amp_bound_graph = 10
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()

    ## Modify data so that outlying error points are visible at the edges of the plot
    for i in range(len(NEURON_ths)):
        if NEURON_ths[i] <= 10 and model_ths[i] > 10:
            model_ths[i] = 10
        elif model_ths[i] <= 10 and NEURON_ths[i] > 10:
            NEURON_ths[i] = 10

    colors = ["k", "m", "lime", "b"]
    start_vth_itr = 0
    for i in range(len(points_per_pathway)):
        if i == len(points_per_pathway):
            end_vth_itr = len(NEURON_ths)
        else:
            end_vth_itr = start_vth_itr + points_per_pathway[i]
        ax.scatter(NEURON_ths[start_vth_itr:end_vth_itr], model_ths[start_vth_itr:end_vth_itr],s=20, c=colors[i])
        start_vth_itr += points_per_pathway[i]

    # ax.scatter(NEURON_ths, model_ths,s=5, c="black")
    ax.plot([0,upper_amp_bound_graph], [0,upper_amp_bound_graph],lw=1, c="red")
    ax.set(xlabel='NEURON Vths', ylabel='Model Vths')
    plt.xlim(0,upper_amp_bound_graph)
    plt.ylim(0,upper_amp_bound_graph)
    plt.show()

if PLT_THRESHOLD_HISTOGRAMS == 1:
    upper_x_bound_graph = 10

    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()

    bins = np.arange(0,upper_x_bound_graph+1,1)
    ax.hist(NEURON_ths, bins)
    plt.xlim(0,upper_x_bound_graph)
    plt.show()