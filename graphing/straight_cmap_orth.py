'''
    This script creates graphs of the error between a selected predictor model
    and NEURON results. Each model's results must be passed via json LUTs. This
    script is intended for graphing the results of simulations of axons which
    are straight and orthogonal to the DBS lead.
'''

# Imports
import sys
import numpy as np
from matplotlib import pyplot as plt
import json

NEURON_data_json = sys.argv[1]
model_data_json = sys.argv[2]
pulse_width = float(sys.argv[3]) / 1000

PLT_NEURON_VTHS = 0
PLT_model_VTHS = 0
PLT_ABS_ERROR = 1
PLT_PERCENT_ERROR = 1

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

NEURON_ths = []
model_ths = []

h_arry = []
v_arry = []

for h_dist in h_distances:
    for v_disp in v_displacements:
        NEURON_ths.append(NEURON_data[str(pulse_width)][str(h_dist)][str(v_disp)])
        model_ths.append(model_data[str(pulse_width)][str(h_dist)][str(v_disp)])

        h_arry.append(h_dist + 0.635)
        v_arry.append(v_disp + 6.75)

if PLT_NEURON_VTHS == 1:
    NEURON_ths_cmap = plt.figure(figsize=(7,7))
    e_plot = NEURON_ths_cmap.add_subplot(111)
    c = np.abs(np.reshape(NEURON_ths,len(NEURON_ths))) # Colors of plot determined by voltage (reshapes vector first)
    cmhot = plt.get_cmap("YlOrRd") # Color map to use ("hot")
    caxSelFiber = e_plot.scatter(h_arry, v_arry, s=30, c=c, cmap=cmhot) # Scatterplot of points
    plt.axis('scaled')
    e_plot.set_xlabel('X [mm]')
    e_plot.set_ylabel('Y [mm]')
    e_plot.set_xlim(-1.135,7.335)
    e_plot.set_ylim(1.75,11.75)
    e_plot.set_title('NEURON Predicted Vths')
    cbar = plt.colorbar(caxSelFiber)
    cbar.ax.set_title('V', rotation=0)
    plt.show()

if PLT_model_VTHS == 1:
    model_ths_cmap = plt.figure(figsize=(7,7))
    e_plot = model_ths_cmap.add_subplot(111)
    c = np.abs(np.reshape(model_ths,len(model_ths))) # Colors of plot determined by voltage (reshapes vector first)
    cmhot = plt.get_cmap("YlOrRd") # Color map to use ("hot")
    caxSelFiber = e_plot.scatter(h_arry, v_arry, s=30, c=c, cmap=cmhot) # Scatterplot of points
    plt.axis('scaled')
    e_plot.set_xlabel('X [mm]')
    e_plot.set_ylabel('Y [mm]')
    e_plot.set_xlim(-1.135,7.335)
    e_plot.set_ylim(1.75,11.75)
    e_plot.set_title('Model Predicted Vths')
    cbar = plt.colorbar(caxSelFiber)
    cbar.ax.set_title('V', rotation=0)
    plt.show()

if PLT_ABS_ERROR == 1:
    model_abs_error_vths = np.subtract(NEURON_ths, model_ths)

    model_abs_error_ths_cmap = plt.figure(figsize=(7,7))
    e_plot = model_abs_error_ths_cmap.add_subplot(111)
    c = np.absolute(np.reshape(model_abs_error_vths,len(model_abs_error_vths))) # Colors of plot determined by voltage (reshapes vector first)
    cmhot = plt.get_cmap("YlOrRd") # Color map to use ("hot")
    caxSelFiber = e_plot.scatter(h_arry, v_arry, s=30, c=c, cmap=cmhot) # Scatterplot of points
    plt.axis('scaled')
    e_plot.set_xlabel('X [mm]')
    e_plot.set_ylabel('Y [mm]')
    e_plot.set_xlim(-1.135,7.335)
    e_plot.set_ylim(1.75,11.75)
    e_plot.set_title('Model Vths Absolute Error')
    cbar = plt.colorbar(caxSelFiber)
    cbar.ax.set_title('V', rotation=0)
    plt.show()

if PLT_PERCENT_ERROR == 1:
    model_percent_error_vths = np.multiply(np.divide(np.subtract(NEURON_ths, model_ths), NEURON_ths), 100)

    model_percent_error_ths_cmap = plt.figure(figsize=(7,7))
    e_plot = model_percent_error_ths_cmap.add_subplot(111)
    c = np.absolute(np.reshape(model_percent_error_vths,len(model_percent_error_vths))) # Colors of plot determined by voltage (reshapes vector first)
    cmhot = plt.get_cmap("YlOrRd") # Color map to use ("hot")
    caxSelFiber = e_plot.scatter(h_arry, v_arry, s=30, c=c, cmap=cmhot) # Scatterplot of points
    plt.axis('scaled')
    e_plot.set_xlabel('X [mm]')
    e_plot.set_ylabel('Y [mm]')
    e_plot.set_xlim(-1.135,7.335)
    e_plot.set_ylim(1.75,11.75)
    e_plot.set_title('Model Vths Percent Error')
    cbar = plt.colorbar(caxSelFiber)
    cbar.ax.set_title('%', rotation=0)
    plt.show()
