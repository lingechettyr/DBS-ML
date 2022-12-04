'''
    This script creates recruitment curve plots for the predictions of 
    NEURON and a predictor model. The results must be passed in via json
    LUTs. Recruitment curves and/or error boxplots can be graphed. Pulse 
    widths of interest can be specified.
'''

from matplotlib import pyplot as plt
import numpy as np
import sys
import json
import pandas as pd

#pulse_widths = [60,120,200,300,500]
#pulse_widths = [75,135,175,275,450]
#pulse_widths = [60, 75, 90, 105, 120, 135, 150, 175, 200, 225, 250, 275, 300, 350, 400, 450, 500]
pulse_widths = [75, 135, 450]
sweep_resolution = 100
test_stim_amps = np.linspace(0,10,sweep_resolution)

PLT_CURVES = 1

def getPercentActivation(amplitude, NEURON_ths, model_ths):
    NEURON_act_cnt = 0
    model_act_cnt = 0

    for ind in range(len(NEURON_ths)):
        if amplitude >= NEURON_ths[ind]:
            NEURON_act_cnt += 1
        if amplitude >= model_ths[ind]:
            model_act_cnt += 1  

    NEURON_perc_act = 100 * NEURON_act_cnt / len(NEURON_ths)
    model_perc_act = 100 * model_act_cnt / len(model_ths)

    return NEURON_perc_act, model_perc_act

NEURON_acts_all = []
model_acts_all = []

cmd_line_itr = 1
exitCond = False
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

        NEURON_act_pw = np.empty((len(pulse_widths), (len(test_stim_amps))))
        model_act_pw = np.empty((len(pulse_widths), (len(test_stim_amps))))

        for pw_ind in range(len(pulse_widths)):

            NEURON_ths = []
            model_ths = []

            full_range = range(len(model_data[str(pulse_widths[pw_ind]/1000)]))

            for dict_ind in full_range:
                if dict_ind in model_data["problem_inds"]:
                    continue
                
                NEURON_ths.append(NEURON_data[str(pulse_widths[pw_ind]/1000)][str(dict_ind)])
                model_ths.append(model_data[str(pulse_widths[pw_ind]/1000)][str(dict_ind)])

            NEURON_acts = []
            model_acts = []
            model_acts_error = np.empty(0)

            for amp in test_stim_amps:
                NEURON_act, model_act = getPercentActivation(amp, NEURON_ths, model_ths)
                NEURON_acts.append(NEURON_act)
                model_acts.append(model_act)
                model_acts_error = np.append(model_acts_error, (model_act-NEURON_act))
            
            print("Number of thresholds: " + str(len(NEURON_ths)))
            print("Model average error: " + str(np.average(np.absolute(model_acts_error))) + "%")

            NEURON_act_pw[pw_ind] = NEURON_acts
            model_act_pw[pw_ind] = model_acts

        NEURON_acts_all.append(NEURON_act_pw)
        model_acts_all.append(model_act_pw)
    
        cmd_line_itr += 2
    except:
        exitCond = True

colors = ["m", "lime", "b"]
if PLT_CURVES == 1:
    for pw_ind in range(len(pulse_widths)):

        fig = plt.figure(figsize=(7,7))
        ax = plt.subplot()
        for i in range(len(NEURON_acts_all)):
            ax.plot(test_stim_amps, NEURON_acts_all[i][pw_ind], '-k', lw=3, label='NEURON')
            ax.plot(test_stim_amps, model_acts_all[i][pw_ind], colors[i], lw=2, label='model')
        title_str = "Pulse width = " + str(int(pulse_widths[pw_ind])) + " us"
        ax.set(title=title_str, xlabel='Stimulus Amplitude (V)', ylabel='Activation (%)')
        ax.set_ylim(0,100)
        ax.set_xlim(0,10)
        #ax.legend()
        ax.grid()
        plt.show()

