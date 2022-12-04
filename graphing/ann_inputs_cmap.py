'''
    This script creates a 3d heat map of the validation dataset accuracy
    as a function of the sizes of the 3 spatial input arrays. The hyperparameter
    search results must be passed in via csv file.
'''

import numpy as np
import pandas as pd
import sys
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

input_csv = sys.argv[1]

acc_threshold = 80

df = pd.read_csv(input_csv)
num_cols = len(df.iloc[0])
num_rows = len(df)

ecs = [0,1,3,5,7,9,11]
fsds = [0,1,3,5,7,9,11]
ssds = [0,1,3,5,7,9,11]

matrix = np.zeros(shape=(len(ecs),len(fsds),len(ssds)))
matrix = np.multiply(matrix, 0.5)

for i in range(num_rows):
    num_ecs = int(df.iloc[i][0])
    num_fsds = int(df.iloc[i][1])
    num_ssds = int(df.iloc[i][2])
    acc = 100 - float(df.iloc[i][3])

    current_acc = matrix[ecs.index(num_ecs), fsds.index(num_fsds), ssds.index(num_ssds)]
    if acc > acc_threshold:
        matrix[ecs.index(num_ecs), fsds.index(num_fsds), ssds.index(num_ssds)] = acc

ecs_ax = []
fsds_ax = []
ssds_ax = []
acc_flat = []

for i in range(len(ecs)):
    for j in range(len(fsds)):
        for k in range(len(ssds)):
            acc = matrix[i,j,k]
            if acc > 0:
                ecs_ax.append(ecs[i])
                fsds_ax.append(fsds[j])
                ssds_ax.append(ssds[k])
                acc_flat.append(acc)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
pnt3d = ax.scatter(ecs_ax, fsds_ax, ssds_ax, s=30, c= acc_flat, cmap='winter')
cbar=plt.colorbar(pnt3d, pad=0.2)
#cbar.set_clim(acc_threshold,1)
cbar.set_label("Validation Accuracy")
ax.set_xlabel('Vs')
ax.set_ylabel('FSDs')
ax.set_zlabel('SSDs')
plt.show()