'''
    This script generates a population of pseudo fiber trajectories across 3d space. It utilizes
    a 'moving origin' approach and the spherical coordinate system to make the geometric computations
    easier.

    NOTE: Using the spherical coordinate system causes problems when trying to draw a random angular 
    displacement from a bivariate normal distribution, specifically when the distribution is centered 
    at or near the the poles. The resulting tracts were biased towards the (vertical) poles of the 
    system. I found a workaround to this by always using distributions centered at the equator, and 
    then temporarily tranforming the coordinate system so the sampled points are in the desired place.
'''

import sys
import math
import random
import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from mayavi import mlab

num_fibers = int(sys.argv[1])
write_to_file = int(sys.argv[2])
if write_to_file == 1:
    output_filename = sys.argv[3]

origin = [0,0,6.75] # origin coordinates [x,y,z]
bound_length = 28
bounds_x = [origin[0]-bound_length, origin[0]+bound_length]
bounds_y = [origin[1]-bound_length, origin[1]+bound_length]
bounds_z = [origin[2]-bound_length, origin[2]+bound_length]
point_to_point = 0.5

bound_points_x = [min(bounds_x), min(bounds_x), min(bounds_x), min(bounds_x),
                  max(bounds_x), max(bounds_x), max(bounds_x), max(bounds_x)]
bound_points_y = [min(bounds_y), min(bounds_y), max(bounds_y), max(bounds_y),
                  min(bounds_y), min(bounds_y), max(bounds_y), max(bounds_y)]
bound_points_z = [min(bounds_z), max(bounds_z), min(bounds_z), max(bounds_z),
                  min(bounds_z), max(bounds_z), min(bounds_z), max(bounds_z)]

arc_dist_size = 30 # in degrees
arc_dist_size_rad_half = (arc_dist_size * math.pi / 2) / 180

def CartToSpherical(x, y, z):
    radius = point_to_point #math.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / point_to_point)
    if x > 0:
        phi = np.arctan(y/x)
    elif x < 0 and y >= 0: 
        phi = np.arctan(y/x) + np.pi
    elif x < 0 and y < 0: 
        phi = np.arctan(y/x) - np.pi
    elif x == 0 and y > 0: 
        phi = np.pi / 2
    elif x == 0 and y < 0: 
        phi =  -1 * np.pi / 2
    else:
        raise Exception("Undefined phi value for x = " + str(x) + ", y = " + str(y) + ", z = " + str(z))
    
    return radius, theta, phi

def SphericalToCart(radius, theta, phi):
    x = radius * math.cos(phi) * math.sin(theta)
    y = radius * math.sin(phi) * math.sin(theta)
    z = radius * math.cos(theta)
    return x, y, z

point_arry_x = []
point_arry_y = []
point_arry_z = []

start_radius_upper_bound = 12
start_radius_lower_bound =(1.27/2) + 0.5 #radius of the lead added to thickness of encapsulation tissue

for i in range(num_fibers):
    rand_radius = random.uniform(start_radius_lower_bound, start_radius_upper_bound)

    ## 6/24 Attempt to produce a uniform distribution IN THE THRESHOLDS of the artificial datasets
    # rand_radius = start_radius_upper_bound * (1 - np.random.exponential(.2, 1)[0])
    # while(rand_radius <= start_radius_lower_bound or rand_radius >= start_radius_upper_bound):
    #     rand_radius = start_radius_upper_bound * (1 - np.random.exponential(.2, 1)[0])

    rand_theta_temp = random.uniform(0,1)
    rand_theta = math.acos((2*rand_theta_temp)-1)
    rand_phi = random.uniform(-1*np.pi, np.pi)
    x_p1_temp, y_p1_temp, z_p1_temp = SphericalToCart(rand_radius, rand_theta, rand_phi)

    starting_point = [x_p1_temp + origin[0], y_p1_temp + origin[1], z_p1_temp + origin[2]]

    rand_theta_temp = random.uniform(0,1)
    rand_theta = math.acos((2*rand_theta_temp)-1)
    rand_phi = random.uniform(-1*np.pi, np.pi)
    x_p1_temp, y_p1_temp, z_p1_temp = SphericalToCart(point_to_point, rand_theta, rand_phi)

    prior_point_1 = [x_p1_temp + starting_point[0], y_p1_temp + starting_point[1], z_p1_temp + starting_point[2]]

    current_point_1 = starting_point
    current_point_2 = starting_point

    points_x = []
    points_y = []
    points_z = []

    points_x.append(current_point_1[0])
    points_y.append(current_point_1[1])
    points_z.append(current_point_1[2])

    # Start building fiber in one direction
    while current_point_1[0] >= min(bounds_x) and current_point_1[0] <= max(bounds_x) and current_point_1[1] >= min(bounds_y) and current_point_1[1] <= max(bounds_y) and current_point_1[2] >= min(bounds_z) and current_point_1[2] <= max(bounds_z):

        # Treat current point as origin to make polar conversion easy
        # Subtract current point cartesian coordinates from prior point
        prior_point_temp = [prior_point_1[0]-current_point_1[0], prior_point_1[1]-current_point_1[1], prior_point_1[2]-current_point_1[2]]
        prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1], -1*prior_point_temp[2]]

        _, next_theta_base, next_phi_base = CartToSpherical(prior_point_opp[0], prior_point_opp[1], prior_point_opp[2])

        ## Naive approach, errors arise at/near the poles
        # next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
        # next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)

        ## FIX: use coordinate transformation matrix multiplication to always get perfect bivariate angular displacement distributions 
        new_base_theta = np.pi/2 # generate the distribution of points around the x axis, 
                                    # which is at the equator in spherical coordinate system
        new_base_phi = 0
        next_theta_pre = np.random.normal(new_base_theta, arc_dist_size_rad_half/2)
        next_phi_pre = np.random.normal(new_base_phi, arc_dist_size_rad_half/2)

        next_x_pre, next_y_pre, next_z_pre = SphericalToCart(point_to_point, next_theta_pre, next_phi_pre)
        next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta_base, next_phi_base)

        # Transform coordinate system so that norm distribution axis (x-axis) is now at the desired vector orientation
        vectorx = [next_x,next_y,next_z]
        uvx = vectorx / np.linalg.norm(vectorx)
        uvz = np.cross(uvx,[0,1,0]) / np.linalg.norm(np.cross(uvx,[0,1,0]))
        uvy = np.cross(uvx,uvz) / np.linalg.norm(np.cross(uvx,uvz))

        rotation_matrix = np.matrix([[uvx[0],uvy[0],uvz[0],0],
                                    [uvx[1],uvy[1],uvz[1],0],
                                    [uvx[2],uvy[2],uvz[2],0],
                                    [0     ,0     ,0     ,1]])

        point = [next_x_pre, next_y_pre, next_z_pre, 1]
        rot_point = np.matmul(rotation_matrix, point)
        next_point_x = rot_point[0,0]
        next_point_y = rot_point[0,1]
        next_point_z = rot_point[0,2]
        
        # Add current point cartesian coordinates back in to get global coordinates for next point
        next_point = [next_point_x + current_point_1[0], next_point_y + current_point_1[1], next_point_z + current_point_1[2]]

        prior_point_1 = current_point_1
        current_point_1 = next_point

        if current_point_1[0] >= min(bounds_x) and current_point_1[0] <= max(bounds_x) and current_point_1[1] >= min(bounds_y) and current_point_1[1] <= max(bounds_y) and current_point_1[2] >= min(bounds_z) and current_point_1[2] <= max(bounds_z):
            points_x.append(current_point_1[0])
            points_y.append(current_point_1[1])
            points_z.append(current_point_1[2])

    prior_point_2 = [points_x[1], points_y[1], points_z[1]]

    # Start building fiber in the opposite direction
    while current_point_2[0] >= min(bounds_x) and current_point_2[0] <= max(bounds_x) and current_point_2[1] >= min(bounds_y) and current_point_2[1] <= max(bounds_y) and current_point_2[2] >= min(bounds_z) and current_point_2[2] <= max(bounds_z):

        # Treat current point as origin to make polar conversion easy
        prior_point_temp = [prior_point_2[0]-current_point_2[0], prior_point_2[1]-current_point_2[1], prior_point_2[2]-current_point_2[2]]
        prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1], -1*prior_point_temp[2]]

        _, next_theta_base, next_phi_base = CartToSpherical(prior_point_opp[0], prior_point_opp[1], prior_point_opp[2])

        ## Naive approach, errors arise at/near the poles
        # next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
        # next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)

        ## FIX: use coordinate transformation matrix multiplication to always get perfect bivariate angular displacement distributions 
        new_base_theta = np.pi/2 # generate the distribution of points around the x axis, 
                                    # which is at the equator in spherical coordinate system
        new_base_phi = 0
        next_theta_pre = np.random.normal(new_base_theta, arc_dist_size_rad_half/2)
        next_phi_pre = np.random.normal(new_base_phi, arc_dist_size_rad_half/2)

        next_x_pre, next_y_pre, next_z_pre = SphericalToCart(point_to_point, next_theta_pre, next_phi_pre)
        next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta_base, next_phi_base)

        # Transform coordinate system so that norm distribution axis (x-axis) is now at the desired vector orientation
        vectorx = [next_x,next_y,next_z]
        uvx = vectorx / np.linalg.norm(vectorx)
        uvz = np.cross(uvx,[0,1,0]) / np.linalg.norm(np.cross(uvx,[0,1,0]))
        uvy = np.cross(uvx,uvz) / np.linalg.norm(np.cross(uvx,uvz))

        rotation_matrix = np.matrix([[uvx[0],uvy[0],uvz[0],0],
                                    [uvx[1],uvy[1],uvz[1],0],
                                    [uvx[2],uvy[2],uvz[2],0],
                                    [0     ,0     ,0     ,1]])

        point = [next_x_pre, next_y_pre, next_z_pre, 1]
        rot_point = np.matmul(rotation_matrix, point)
        next_point_x = rot_point[0,0]
        next_point_y = rot_point[0,1]
        next_point_z = rot_point[0,2]

        next_point = [next_point_x + current_point_2[0], next_point_y + current_point_2[1], next_point_z + current_point_2[2]]

        prior_point_2 = current_point_2
        current_point_2 = next_point

        if current_point_2[0] >= min(bounds_x) and current_point_2[0] <= max(bounds_x) and current_point_2[1] >= min(bounds_y) and current_point_2[1] <= max(bounds_y) and current_point_2[2] >= min(bounds_z) and current_point_2[2] <= max(bounds_z):
            points_x.insert(0,current_point_2[0])
            points_y.insert(0,current_point_2[1])
            points_z.insert(0,current_point_2[2])

    point_arry_x.append(points_x)
    point_arry_y.append(points_y)
    point_arry_z.append(points_z)


mlab.figure(bgcolor=(1,1,1),size=(800,800))

mlab.points3d(bound_points_x, bound_points_y, bound_points_z, color = (0,0,0), scale_factor=1, opacity=1)
mlab.points3d(origin[0], origin[1], origin[2], color = (1,0,0), scale_factor=1, opacity=1)
origin
for i in range(len(point_arry_x)):
    mlab.plot3d(point_arry_x[i], point_arry_y[i], point_arry_z[i], color = (0.2,0.2,0.2), tube_radius = .05)
    #mlab.points3d(point_arry_x[i], point_arry_y[i], point_arry_z[i], color = (1,0,0), scale_factor=0.2, opacity=1)

mlab.show()

if write_to_file == 1:
    with open(output_filename, 'w') as f:
        for i in range(len(point_arry_x)):
            line = ''
            for j in range(len(point_arry_x[i])):
                line += str(point_arry_x[i][j])
                line += ' '
                line += str(point_arry_y[i][j])
                line += ' '
                line += str(point_arry_z[i][j])
                line += ' '
            f.write(line)
            f.write('\n')