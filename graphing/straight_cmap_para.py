'''
    This script creates graphs of the error between a selected predictor model
    and NEURON results. Each model's results must be passed via json LUTs. This
    script is intended for graphing the results of simulations of axons which
    are straight and parallel to the DBS lead.
'''

# Imports
import sys
import numpy as np
from matplotlib import pyplot as plt
import json
from matplotlib.collections import LineCollection

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
v_displacements = [0.0]

h_distances = [round(k,1) for k in h_distances]
v_displacements = [round(k,1) for k in v_displacements]

NEURON_ths = []
model_ths = []

h_arry = []
v_arry = []

line = np.arange(-16.25, 23.4, 0.5)

for h_dist in h_distances:
    for v_disp in v_displacements:
        NEURON_ths.append(NEURON_data[str(pulse_width)][str(h_dist)][str(v_disp)])
        model_ths.append(model_data[str(pulse_width)][str(h_dist)][str(v_disp)])

        h_arry.append((h_dist + 0.635) * np.ones(len(line)))
        v_arry.append(line)

# sourced from https://stackoverflow.com/questions/38208700/matplotlib-plot-lines-with-colors-through-colormap
def multiline(xs, ys, c, ax=None, **kwargs):
    """Plot lines with different colorings

    Parameters
    ----------
    xs : iterable container of x coordinates
    ys : iterable container of y coordinates
    c : iterable container of numbers mapped to colormap
    ax (optional): Axes to plot on.
    kwargs (optional): passed to LineCollection

    Notes:
        len(xs) == len(ys) == len(c) is the number of line segments
        len(xs[i]) == len(ys[i]) is the number of points for each line (indexed by i)

    Returns
    -------
    lc : LineCollection instance.
    """

    # find axes
    ax = plt.gca() if ax is None else ax

    # create LineCollection
    segments = [np.column_stack([x, y]) for x, y in zip(xs, ys)]
    lc = LineCollection(segments, **kwargs)

    # set coloring of line segments
    #    Note: I get an error if I pass c as a list here... not sure why.
    lc.set_array(np.asarray(c))

    # add lines to axes and rescale 
    #    Note: adding a collection doesn't autoscalee xlim/ylim
    ax.add_collection(lc)
    ax.autoscale()
    return lc

if PLT_NEURON_VTHS == 1:
    fig, ax = plt.subplots(figsize=(7,7))
    lc = multiline(h_arry, v_arry, np.absolute(NEURON_ths), cmap='YlOrRd', lw=3)
    axcb = fig.colorbar(lc)
    axcb.ax.set_title('V', rotation=0)
    plt.axis('scaled')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_xlim(-1.135,7.335)
    ax.set_ylim(1.75,11.75)
    ax.set_title('NEURON Predicted Vths')
    plt.show()

if PLT_model_VTHS == 1:
    fig, ax = plt.subplots(figsize=(7,7))
    lc = multiline(h_arry, v_arry, np.absolute(model_ths), cmap='YlOrRd', lw=3)
    axcb = fig.colorbar(lc)
    axcb.ax.set_title('V', rotation=0)
    plt.axis('scaled')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_xlim(-1.135,7.335)
    ax.set_ylim(1.75,11.75)
    ax.set_title('Model Predicted Vths')
    plt.show()

if PLT_ABS_ERROR == 1:
    model_abs_error_vths = np.subtract(NEURON_ths, model_ths)

    fig, ax = plt.subplots(figsize=(7,7))
    lc = multiline(h_arry, v_arry, np.absolute(model_abs_error_vths), cmap='YlOrRd', lw=3)
    axcb = fig.colorbar(lc)
    axcb.ax.set_title('V', rotation=0)
    plt.axis('scaled')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_xlim(-1.135,7.335)
    ax.set_ylim(1.75,11.75)
    ax.set_title('Model Vths Absolute Error')
    plt.show()

if PLT_PERCENT_ERROR == 1:
    model_percent_error_vths = np.multiply(np.divide(np.subtract(NEURON_ths, model_ths), NEURON_ths), 100)

    fig, ax = plt.subplots(figsize=(7,7))
    lc = multiline(h_arry, v_arry, np.absolute(model_percent_error_vths), cmap='YlOrRd', lw=3)
    axcb = fig.colorbar(lc)
    axcb.ax.set_title('%', rotation=0)
    plt.axis('scaled')
    ax.set_xlabel('X [mm]')
    ax.set_ylabel('Y [mm]')
    ax.set_xlim(-1.135,7.335)
    ax.set_ylim(1.75,11.75)
    ax.set_title('Model Vths Percent Error')
    plt.show()
