'''
    This script plots error boxplots for the predicted thresholds of the MRG model
    and a selected predictor model. The predictions must be passed via json LUTs. 
    Only the thresholds of a single pulse-width can be plotted at a time. The user
    selects the pulse-width of interest on the command line.
'''

from matplotlib import pyplot as plt
import numpy as np
import sys
import json
import pandas as pd

NEURON_data_json = sys.argv[1]
model_data_json = sys.argv[2]
pulse_width = float(sys.argv[3]) / 1000

PLT_BOXPLOT_ABS_ERROR = 1
PLT_BOXPLOT_ABS_ERROR_OUTLIERS = 1
PLT_BOXPLOT_PERCENT_ERROR = 1
PLT_BOXPLOT_PERCENT_ERROR_OUTLIERS = 1

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
good_fib_inds = []

for fib_ind in range(num_total_fibers):
    if fib_ind not in model_problem_inds and NEURON_data[str(pulse_width)][str(fib_ind)] < 50:
        NEURON_ths.append(NEURON_data[str(pulse_width)][str(fib_ind)])
        model_ths.append(model_data[str(pulse_width)][str(fib_ind)])
        good_fib_inds.append(fib_ind)

model_abs_error = np.subtract(model_ths, NEURON_ths)
model_perc_error = np.multiply(np.divide(model_abs_error, NEURON_ths), 100)

pulse_widths = [pulse_width*1000]

model_pw_df = pd.DataFrame(np.matrix.transpose(model_abs_error), columns=pulse_widths)
model_pw_df_perc = pd.DataFrame(np.matrix.transpose(model_perc_error), columns=pulse_widths)

print("Number of thresholds within range (0 - 50V): " + str(len(model_ths)))

if PLT_BOXPLOT_ABS_ERROR == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    # boxplot_dict = ax.boxplot(model_pw_df,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,whis=100,medianprops=medianprops, showcaps=False)
    boxplot_dict = ax.boxplot(model_pw_df,labels=pulse_widths,showfliers=False,widths=0.65,manage_ticks=True,medianprops=medianprops, showcaps=False)
    plt.xticks(fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (V)', title='Model Predicted Threshold Error')
    plt.show()


if PLT_BOXPLOT_ABS_ERROR_OUTLIERS == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    # boxplot_dict = ax.boxplot(model_pw_df,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,whis=100,medianprops=medianprops, showcaps=False)
    boxplot_dict = ax.boxplot(model_pw_df,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,medianprops=medianprops, showcaps=False)
    plt.xticks(fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (V)', title='Model Predicted Threshold Error')
    plt.show()

    print()
    fliers = boxplot_dict['fliers']
    for j in range(len(fliers)):
        yfliers = boxplot_dict['fliers'][j].get_ydata()
        xfliers = boxplot_dict['fliers'][j].get_xdata()
        print("Number of voltage error outliers for pw = " + str(pulse_widths[j]) + " us: " + str(len(xfliers)))


if PLT_BOXPLOT_PERCENT_ERROR == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    # boxplot_dict = ax.boxplot(model_pw_df_perc,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,whis=100,medianprops=medianprops, showcaps=False)
    boxplot_dict = ax.boxplot(model_pw_df_perc,labels=pulse_widths,showfliers=False,widths=0.65,manage_ticks=True,medianprops=medianprops, showcaps=False)
    plt.xticks(fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (%)', title='Model Predicted Threshold Error (%)')
    plt.show()


if PLT_BOXPLOT_PERCENT_ERROR_OUTLIERS == 1:
    fig = plt.figure(figsize=(7,7))
    ax = plt.subplot()
    medianprops = dict(linestyle='-', linewidth=2.5, color='red')
    # boxplot_dict = ax.boxplot(model_pw_df_perc,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,whis=100,medianprops=medianprops, showcaps=False)
    boxplot_dict = ax.boxplot(model_pw_df_perc,labels=pulse_widths,showfliers=True,widths=0.65,manage_ticks=True,medianprops=medianprops, showcaps=False)
    plt.xticks(fontsize=12)
    plt.xticks(rotation=45)
    plt.yticks(fontsize=12)
    ax.set(xlabel='Pulse Width (us)', ylabel='Threshold Error (%)', title='Model Predicted Threshold Error (%)')
    plt.show()

    print()
    fliers = boxplot_dict['fliers']
    for j in range(len(fliers)):
        yfliers = boxplot_dict['fliers'][j].get_ydata()
        xfliers = boxplot_dict['fliers'][j].get_xdata()
        print("Number of voltage error outliers for pw = " + str(pulse_widths[j]) + " us: " + str(len(xfliers)))
