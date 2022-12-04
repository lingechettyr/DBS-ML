'''
    This file contains a class, 'FEMgrid', that allows the user to import
    exported data from COMSOL, in spreadsheet format, and convert it into
    a python object (regulargridinterpolator).
'''

import numpy as np
from scipy.interpolate import interp1d
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CubicSpline

import time

class FEMgrid:
    grid_e1 = None

    def __init__(self, filename):
        start_time = time.time()
        f1 = open(filename , mode='r')

        x_coords = []
        y_coords = []
        z_coords = []
        potentials = []

        x_val_prev = None
        y_val_prev = None
        z_val_prev = None

        for line in f1: # Iterate through input text file, save data to python lists
            if line.startswith("%"):
                continue # Skip if line doesn't contain data
            else:
                first, second, third, fourth = line.split()
                x_val = round(float(first), 3)
                y_val = round(float(second), 3)
                z_val = round(float(third), 3)
                potential = round(float(fourth), 10)

                potentials.append(potential)

                if x_val != x_val_prev and x_val not in x_coords:
                    x_coords.append(x_val)
                if y_val != y_val_prev and y_val not in y_coords:
                    y_coords.append(y_val)
                if z_val != z_val_prev and z_val not in z_coords:
                    z_coords.append(z_val)

        f1.close()
    
        self.fem_bounds = [[min(x_coords), max(x_coords)],
                           [min(y_coords), max(y_coords)],
                           [min(z_coords), max(z_coords)]]

        potentials_3d = np.zeros(shape=(len(x_coords),len(y_coords),len(z_coords))) # Initialize array to hold electric potential values
        itr = 0
        for z in range(len(z_coords)): # Iterate through every depth plane (z values)
            for y in range(len(y_coords)): # Iterate through every vertical plane (y values)
                for x in range(len(x_coords)): # Iterate through every lateral plane (x values)
                    potentials_3d[x][y][z] = potentials[itr]*1000# Fills in potential values (values in excel are in volts, must be millivolts)
                    itr += 1 
 
        self.grid_e1 = RegularGridInterpolator([x_coords,y_coords,z_coords], potentials_3d, method = 'linear', fill_value=0) #interpolate potential for each node
        end_time = time.time()
        print("FEM processing took " + str(end_time-start_time) + " s")

    def get3dGrid(self):
        return self.grid_e1

    def getFEMBounds(self):
        return self.fem_bounds