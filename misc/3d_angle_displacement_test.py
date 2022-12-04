'''
    This script generates a population of pseudo fiber trajectories across 3d space. It utilizes
    a 'moving origin' approach and the spherical coordinate system to make the geometric computations
    easier.
'''

import sys
import math
import random
import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev
from mayavi import mlab

new = int(sys.argv[1])
uniform = int(sys.argv[2])
num_points = int(sys.argv[3])
rand_theta_temp = int(sys.argv[4])
rand_phi_temp = int(sys.argv[5])

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

x_p1_temp, y_p1_temp, z_p1_temp = 0, 0, 0

starting_point = [x_p1_temp + origin[0], y_p1_temp + origin[1], z_p1_temp + origin[2]]

# rand_theta_temp = 90
rand_theta = rand_theta_temp * np.pi / 180
# rand_phi_temp = 270
rand_phi = rand_phi_temp * np.pi / 180
x_p1_temp, y_p1_temp, z_p1_temp = SphericalToCart(point_to_point, rand_theta, rand_phi)

prior_point_1 = [x_p1_temp + starting_point[0], y_p1_temp + starting_point[1], z_p1_temp + starting_point[2]]

current_point_1 = starting_point

points_x = []
points_y = []
points_z = []

# points_x.append(current_point_1[0])
# points_y.append(current_point_1[1])
# points_z.append(current_point_1[2])

# Treat current point as origin to make polar conversion easy
prior_point_temp = [prior_point_1[0]-current_point_1[0], prior_point_1[1]-current_point_1[1], prior_point_1[2]-current_point_1[2]]
prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1], -1*prior_point_temp[2]]

_, next_theta_base, next_phi_base = CartToSpherical(prior_point_opp[0], prior_point_opp[1], prior_point_opp[2])

points_x_base = []
points_y_base = []
points_z_base = []

points_x_base.append(current_point_1[0])
points_y_base.append(current_point_1[1])
points_z_base.append(current_point_1[2])

points_x_base.append(prior_point_opp[0]+current_point_1[0])
points_y_base.append(prior_point_opp[1]+current_point_1[1])
points_z_base.append(prior_point_opp[2]+current_point_1[2])

points_x_alt = []
points_y_alt = []
points_z_alt = []

for i in range(num_points):

    if new == 0:
        if uniform == 1:
            next_theta = random.uniform(next_theta_base-arc_dist_size_rad_half, next_theta_base+arc_dist_size_rad_half)
            next_phi = random.uniform(next_phi_base - arc_dist_size_rad_half, next_phi_base + arc_dist_size_rad_half)
        else:
            next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
            next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)
    else:
        
        ## Naive approach, errors arise at/near the poles
        next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
        next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)

        ## FIX ATTEMPT: use coordinate transformation matrix multiplication, ALMOST PASSES, troubles at the poles still
        new_base_theta = np.pi/2 # generate the distribution of points around the x axis, 
                                    # which is at the equator in spherical coordinate system
        new_base_phi = 0

        if uniform == 1:
            next_theta_pre = np.random.uniform(new_base_theta-arc_dist_size_rad_half, new_base_theta+arc_dist_size_rad_half)
            next_phi_pre = np.random.uniform(new_base_phi-arc_dist_size_rad_half, new_base_phi+arc_dist_size_rad_half)
        else:
            next_theta_pre = np.random.normal(new_base_theta, arc_dist_size_rad_half/2)
            next_phi_pre = np.random.normal(new_base_phi, arc_dist_size_rad_half/2)

        next_x_pre, next_y_pre, next_z_pre = SphericalToCart(point_to_point, next_theta_pre, next_phi_pre)
        ## Potential ERROR at the below line, changed to use the base theta and phi from non-base, what did this do exactly? FIXME
        next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta_base, next_phi_base)
        #next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta, next_phi)

        next_point_alt = [next_x_pre + current_point_1[0], next_y_pre + current_point_1[1], next_z_pre + current_point_1[2]]

        points_x_alt.append(next_point_alt[0])
        points_y_alt.append(next_point_alt[1])
        points_z_alt.append(next_point_alt[2])

        # Transform coordinate system so that norm distribution axis (x-axis) is now at the desired vector orientation
        vectorx = [next_x,next_y,next_z]
        uvx = vectorx / np.linalg.norm(vectorx)
        uvz = np.cross(uvx,[0,1,0]) / np.linalg.norm(np.cross(uvx,[0,1,0]))
        uvy = np.cross(uvx,uvz) / np.linalg.norm(np.cross(uvx,uvz))

        rotation_matrix = np.matrix([[uvx[0],uvy[0],uvz[0],0],
                                    [uvx[1],uvy[1],uvz[1],0],
                                    [uvx[2],uvy[2],uvz[2],0],
                                    [0     ,0     ,0     ,1]])

        # rotation_matrix = np.matrix([[uvx[0],uvx[1],uvx[2],0],
        #                             [uvy[0],uvy[1],uvy[2],0],
        #                             [uvz[0],uvx[1],uvz[2],0],
        #                             [0     ,0     ,0     ,1]])

        point = [next_x_pre, next_y_pre, next_z_pre, 1]
        rot_point = np.matmul(rotation_matrix, point)
        next_point_x = rot_point[0,0]
        next_point_y = rot_point[0,1]
        next_point_z = rot_point[0,2]
    
    if new == 0:
        next_point_x, next_point_y, next_point_z = SphericalToCart(point_to_point, next_theta, next_phi)

    next_point = [next_point_x + current_point_1[0], next_point_y + current_point_1[1], next_point_z + current_point_1[2]]

    points_x.append(next_point[0])
    points_y.append(next_point[1])
    points_z.append(next_point[2])

# print(uvx)
# print(uvy)
# print(uvz)

mlab.figure(bgcolor=(1,1,1),size=(800,800))
mlab.plot3d(points_x_base, points_y_base, points_z_base, color = (1,0,0), tube_radius = .003)
mlab.points3d(points_x_base, points_y_base, points_z_base, color = (0.2,0.2,0.2), scale_factor=0.02, opacity=1)

#mlab.points3d(bound_points_x, bound_points_y, bound_points_z, color = (0,0,0), scale_factor=1, opacity=1)
mlab.points3d(points_x, points_y, points_z, color = (0.2,0.2,0.2), scale_factor=0.02, opacity=1)
mlab.points3d(points_x_alt, points_y_alt, points_z_alt, color = (1,0,0), scale_factor=0.02, opacity=1)

points_x_temp = [0,uvx[0]]
points_y_temp = [0,uvx[1]]
points_z_temp = [0,uvx[2]]

#mlab.plot3d(points_x_temp, points_y_temp, points_z_temp, color = (1,1,0), tube_radius = .003)
# mlab.plot3d([0,0], [0,1], [0,0], color = (0,1,0), tube_radius = 1)
# mlab.plot3d([0,0], [0,0], [0,1], color = (0,0,1), tube_radius = 1)

# # This is not working and really weird
# mlab.plot3d([0.00001,1], [0.00001,0], [0.00001,0], color = (1,0,0), tube_radius = .003)
# mlab.plot3d([0.00001,0], [0,1], [0,0], color = (0,1,0), tube_radius = .003)
# mlab.plot3d([0.00001,0], [0.00001,0], [0,1], color = (0,0,1), tube_radius = .003)

# mlab.plot3d([0.00001,uvx[0]], [0.00001,uvx[1]], [0.00001,uvx[2]], color = (1,1,0), tube_radius = .003)
# mlab.plot3d([0.00001,uvy[0]], [0.00001,uvy[1]], [0.00001,uvy[2]], color = (1,0,1), tube_radius = .003)
# mlab.plot3d([0.00001,uvz[0]], [0.00001,uvz[1]], [0.00001,uvz[2]], color = (0,1,1), tube_radius = .003)

mlab.show()
