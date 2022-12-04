'''
    This is a test script to test a method I've come up with to generate
    pseudo fiber trajectories for ANN training. The method in this script
    generates 2d trajectories, whereas I need to expand it to 3 dimensions.
'''

import math
import random
import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from matplotlib import pyplot as plt

origin = [0,0] # origin coordinates [x,y]
bound_length = 25
bounds_x = [origin[0]-bound_length, origin[0]+bound_length]
bounds_y = [origin[1]-bound_length, origin[1]+bound_length]
point_to_point = 0.5

arc_dist_size = 60 # in degrees
arc_dist_size_rad_half = (arc_dist_size * math.pi / 2) / 180

def CartToPolar(x, y):
    radius = point_to_point #math.sqrt(x**2 + y**2)
    phi = math.atan2(y,x)
    return radius, phi

def PolarToCart(radius, phi):
    x = radius * math.cos(phi)
    y = radius * math.sin(phi)
    return x, y

while(True):
    prior_point = [origin[0], bounds_y[0]]
    current_point = [origin[0], bounds_y[0]+point_to_point]

    points_x = []
    points_y = []

    while current_point[0] >= min(bounds_x) and current_point[0] <= max(bounds_x) and current_point[1] >= min(bounds_y) and current_point[1] <= max(bounds_y):

        # Treat current point as origin to make polar conversion easy
        prior_point_temp = [prior_point[0]-current_point[0], prior_point[1]-current_point[1]]
        prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1]]

        _, next_phi_base = CartToPolar(prior_point_opp[0], prior_point_opp[1])
        
        # next_phi = random.uniform(next_phi_base - arc_dist_size_rad_half, next_phi_base + arc_dist_size_rad_half)
        next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)
        
        next_point_x, next_point_y = PolarToCart(point_to_point, next_phi)
        next_point = [next_point_x + current_point[0], next_point_y + current_point[1]]

        points_x.append(current_point[0])
        points_y.append(current_point[1])

        prior_point = current_point
        current_point = next_point

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(points_x, points_y, '-b', lw=0.75, ms=1)
    plt.axis('scaled')
    plt.show()
    plt.close()