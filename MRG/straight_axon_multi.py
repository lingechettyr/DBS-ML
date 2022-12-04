'''
    This file simulates straight MRG axon models in rows, beginning with the fiber
    closest to the DBS lead. The thresholds are outputted to json files, 
    one for each fiber stimulated. The positions and pulse widths to 
    stimulate at are inputted via a row from a csv.
'''

# Imports
import sys
import numpy as np
import time
from neuron import h, gui
import json
import csv
import os

sys.path.append("../")
from lib.NEURON import axon
from lib.COMSOL import FEM

h.nrn_load_dll("../lib/NEURON/x86_64/.libs/libnrnmech.so")

# COMMAND LINE INPUTS: 
electrode1File = sys.argv[1] 
output_dir = sys.argv[2] 
df_csv = sys.argv[3]
df_index = int(sys.argv[4])
para = int(sys.argv[5]) # 0 if orthogonal fibers, 1 if parallel

number_of_nodes = 41
diameter = 5.7
node_to_node = 0.5
STINmul = 10
compartDiv = STINmul + 5
num_wfs = 1
frequency = 130
shift = 0

# CSV Input Params
with open(df_csv) as fd:
    csv_reader = csv.reader(fd)
    row = [row for idx, row in enumerate(csv_reader) if idx == df_index]

fiber_h_distances = np.arange(int(row[0][0]),int(row[0][1])+1,int(row[0][2]))
fiber_v_displacement_str = row[0][3]
pulse_width_str = row[0][4]

fiber_v_displacement = float(fiber_v_displacement_str) / 1000   
pulse_width = float(pulse_width_str) / 1000

def mkdirp(dir):
    '''make a directory if it doesn't exist'''
    if not os.path.exists(dir):
        os.mkdir(dir)

mkdirp(output_dir)        
  
def getExtracellPotChars():
    maxSSD = -20.0
    maxSSD_node = 0
    maxEC = -20.0
    maxEC_node = 0

    for i in range(1, len(fiberNodeVoltages)-1):
        temp_ssd = -1 * (fiberNodeVoltages[i-1] - (2*fiberNodeVoltages[i]) + fiberNodeVoltages[i+1]) # stimulus is cathodic, so multiply by -1
        secSpatialDiffs.append(temp_ssd)
        if temp_ssd > maxSSD:
            maxSSD = temp_ssd
            maxSSD_node = i

        if fiberNodeVoltages[i] > maxEC:
                maxEC = fiberNodeVoltages[i]
                maxEC_node = i

    return maxSSD, (maxSSD_node - 1), maxEC, maxEC_node 

fem = FEM.FEMgrid(electrode1File)
grid_e1 = fem.get3dGrid()

axon = axon.Axon(diameter,number_of_nodes)
compartmentPos = np.asarray(axon.getCompartmentPositions())

orthogonal_offset = 0 # center of the active contact in the X dimension
parallel_offset = 6.75 # center of the active contact in the Z dimension

if para == 0:
    used_offset = orthogonal_offset
else:
    used_offset = parallel_offset

pos_arry = (compartmentPos/1000)+used_offset-(((number_of_nodes-1)/2)*node_to_node)-0.0005
pos_arry = pos_arry.round(decimals=5)
pos_arry = pos_arry[0:int((number_of_nodes-1)*compartDiv)+1]
fiberLVoltages = np.zeros(len(pos_arry), dtype=float)  #container for voltage data at nodes only    

for fiber_h_distance in fiber_h_distances:

    output_file = output_dir + "/FC_" + str(fiber_h_distance) + "_" + str(fiber_v_displacement_str) + "_" + str(pulse_width_str) + ".json"

    fiber_h_distance = float(fiber_h_distance) / 1000

    if para == 0:
        selectedFiberLx = pos_arry - (shift * 0.5)
        selectedFiberLy = np.ones(len(selectedFiberLx)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLz = np.ones(len(selectedFiberLx)) * (6.75 + fiber_v_displacement)    #specifying nodal locations for a fiber
    else:
        selectedFiberLz = pos_arry - (shift * 0.5)
        selectedFiberLy = np.ones(len(selectedFiberLz)) * (0.635 + fiber_h_distance) # the electrode is along the z-axis at x=0, y=0.4
        selectedFiberLx = np.ones(len(selectedFiberLz)) * 0    #specifying nodal locations for a fiber
           
    # Get compartmental EC potentials from previously made 3d-grid
    for i in range(len(selectedFiberLx)):
        try:
                    
            fiberLVoltages[i]=grid_e1( [selectedFiberLx[i], selectedFiberLy[i], selectedFiberLz[i]] )
                    
        except Exception as e:
            print("WARNING: 3d-position out of COMSOL range!")
            pass
                
    voltages1 = fiberLVoltages

    threshold = axon.findThreshold(voltages1, pulse_width, frequency, num_wfs)

    fiberNodeVoltages = fiberLVoltages[::STINmul + 5] 
    secSpatialDiffs = []
    maxSecSpatDiff, maxSecSpatDiff_node, maxEC, maxEC_node = getExtracellPotChars() # get the nodal extracellular properties: second spatial differences, potentials, etc
    ec_pots = -1 * fiberNodeVoltages # stimulus is cathodic, so multiply by -1
    ec_potentials = ec_pots.tolist()

    # the below outputs all the relevant training data in json format, for use to construct training data jsons

    tg_dict = {}
    tg_dict["Fiber_Properties"] = {}
    tg_dict["Fiber_Properties"]["diameter"] = diameter
    tg_dict["Fiber_Properties"]["internodal_region_cnt"] = STINmul
    tg_dict["Fiber_Properties"]["horizontal_distance"] = fiber_h_distance
    tg_dict["Fiber_Properties"]["vertical_displacement"] = fiber_v_displacement

    tg_dict["Stimulus_Properties"] = {}
    tg_dict["Stimulus_Properties"]["pulse_width"] = pulse_width
    tg_dict["Stimulus_Properties"]["num_wfs"] = num_wfs
    tg_dict["Stimulus_Properties"]["frequency"] = frequency

    tg_dict["Activation_Properties"] = {}
    tg_dict["Activation_Properties"]["threshold_multiplier"] = threshold

    tg_dict["Extracellular_Potential_Properties"] = {}
    tg_dict["Extracellular_Potential_Properties"]["ec_potentials"] = ec_potentials

    with open(output_file, 'w') as outfile:
        json.dump(tg_dict, outfile, indent=4)        
                  
######################################################################################################