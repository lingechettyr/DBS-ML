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
from scipy.interpolate import splprep
from scipy.interpolate import splev

sys.path.append("../")
from lib.DTI import graph_DTI

num_of_points = int(sys.argv[1])

spline_smoothing = .2

origin = [0,0,6.75] # origin coordinates [x,y,z]
bound_length = 10 #28
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

leftLeadPos = [[0,0],[0,0],[0,10]]
# Plotting Code #
def plotLead():
    ## Set up rotation ##
    lead_x = [0,0]
    lead_y = [0,0]
    lead_z = [0,10]

    # Get original DBS lead coordinates
    x_temp = leftLeadPos[0] #[119, 122]
    y_temp = leftLeadPos[1] #[78, 78]
    z_temp = leftLeadPos[2] #[64, 72]

    # First translate original dbs lead position to origin
    x_trans = np.subtract(x_temp,x_temp[0])
    y_trans = np.subtract(y_temp,y_temp[0])
    z_trans = np.subtract(z_temp,z_temp[0])

    # "uva = unit vector a"
    vectorz = [x_trans[1],y_trans[1],z_trans[1]]
    uvz = vectorz / np.linalg.norm(vectorz)
    uvx = np.cross(uvz,[0,1,0])
    uvy = np.cross(uvz,uvx)

    rotation_matrix = np.matrix([[uvx[0],uvx[1],uvx[2],0],
                                [uvy[0],uvy[1],uvy[2],0],
                                [uvz[0],uvz[1],uvz[2],0],
                                [0     ,0     ,0     ,1]])

    lead_color = (0.65, 0.65, 0.65)
    inactive_contact_color = (0.25, 0.25, 0.25)
    active_contact_color = (0.60, 0, 0)

    #mlab.points3d(x_temp,y_temp,z_temp,color=(1,0,0),scale_factor = 1.5)

    # Rounded Tip
    radius = 1.27 / 2
    step = np.pi / 16
    position = -1 * (1.5 - radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi+step,step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    z = np.add(z,position)

    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

    # Remainder of tip
    height = 1.5 - radius
    position = -1 * (1.5 - radius)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

    # First contact
    height = 1.5
    position = (1.5 * 0)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=inactive_contact_color)

    # First gap
    height = 1.5
    position = (1.5 * 1)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

    # Second contact
    height = 1.5
    position = (1.5 * 2)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=inactive_contact_color)

    # Second gap
    height = 1.5
    position = (1.5 * 3)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

    # Third contact
    height = 1.5
    position = (1.5 * 4)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=active_contact_color)

    # Third gap
    height = 1.5
    position = (1.5 * 5)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

    # Fourth contact
    height = 1.5
    position = (1.5 * 6)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=inactive_contact_color)

    # Fourth gap
    height = 7
    position = (1.5 * 7)
    height_step = np.arctan2(height,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,height+position)
    x_rot = np.empty(shape=x.shape)
    y_rot = np.empty(shape=y.shape)
    z_rot = np.empty(shape=z.shape)

    for i, j in np.ndindex(x.shape):
        point = [x[i,j], y[i,j], z[i,j], 1]
        rot_point = np.matmul(rotation_matrix, point)
        x_rot[i,j] = rot_point[0,0]
        y_rot[i,j] = rot_point[0,1]
        z_rot[i,j] = rot_point[0,2]

    x_rot = np.add(x_rot, x_temp[0])
    y_rot = np.add(y_rot, y_temp[0])
    z_rot = np.add(z_rot, z_temp[0])

    mlab.mesh(x_rot, y_rot, z_rot, color=lead_color)

start_radius_upper_bound = 5
start_radius_lower_bound =(1.27/2) + 0.5 #radius of the lead added to thickness of encapsulation tissue

rand_radius = random.uniform(start_radius_lower_bound, start_radius_upper_bound)
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

while current_point_1[0] >= min(bounds_x) and current_point_1[0] <= max(bounds_x) and current_point_1[1] >= min(bounds_y) and current_point_1[1] <= max(bounds_y) and current_point_1[2] >= min(bounds_z) and current_point_1[2] <= max(bounds_z):

    # Treat current point as origin to make polar conversion easy
    prior_point_temp = [prior_point_1[0]-current_point_1[0], prior_point_1[1]-current_point_1[1], prior_point_1[2]-current_point_1[2]]
    prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1], -1*prior_point_temp[2]]

    _, next_theta_base, next_phi_base = CartToSpherical(prior_point_opp[0], prior_point_opp[1], prior_point_opp[2])

    ## Naive approach, errors arise at/near the poles
    next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
    next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)

    ## FIX ATTEMPT: use coordinate transformation matrix multiplication, ALMOST PASSES, troubles at the poles still
    new_base_theta = np.pi/2 # generate the distribution of points around the x axis, 
                                # which is at the equator in spherical coordinate system
    new_base_phi = 0
    next_theta_pre = np.random.normal(new_base_theta, arc_dist_size_rad_half/2)
    next_phi_pre = np.random.normal(new_base_phi, arc_dist_size_rad_half/2)

    next_x_pre, next_y_pre, next_z_pre = SphericalToCart(point_to_point, next_theta_pre, next_phi_pre)
    next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta, next_phi)

    # Transform coordinate system so that norm distribution axis (x-axis) is now at the desired vector orientation
    vectorx = [next_x,next_y,next_z]
    uvx = vectorx / np.linalg.norm(vectorx)
    uvz = np.cross(uvx,[0,1,0])
    uvy = np.cross(uvx,uvz)

    rotation_matrix = np.matrix([[uvx[0],uvy[0],uvz[0],0],
                                [uvx[1],uvy[1],uvz[1],0],
                                [uvx[2],uvy[2],uvz[2],0],
                                [0     ,0     ,0     ,1]])

    point = [next_x_pre, next_y_pre, next_z_pre, 1]
    rot_point = np.matmul(rotation_matrix, point)
    next_point_x = rot_point[0,0]
    next_point_y = rot_point[0,1]
    next_point_z = rot_point[0,2]
    
    next_point = [next_point_x + current_point_1[0], next_point_y + current_point_1[1], next_point_z + current_point_1[2]]

    prior_point_1 = current_point_1
    current_point_1 = next_point

    if current_point_1[0] >= min(bounds_x) and current_point_1[0] <= max(bounds_x) and current_point_1[1] >= min(bounds_y) and current_point_1[1] <= max(bounds_y) and current_point_1[2] >= min(bounds_z) and current_point_1[2] <= max(bounds_z):
        points_x.append(current_point_1[0])
        points_y.append(current_point_1[1])
        points_z.append(current_point_1[2])

prior_point_2 = [points_x[1], points_y[1], points_z[1]]

while current_point_2[0] >= min(bounds_x) and current_point_2[0] <= max(bounds_x) and current_point_2[1] >= min(bounds_y) and current_point_2[1] <= max(bounds_y) and current_point_2[2] >= min(bounds_z) and current_point_2[2] <= max(bounds_z):

    # Treat current point as origin to make polar conversion easy
    prior_point_temp = [prior_point_2[0]-current_point_2[0], prior_point_2[1]-current_point_2[1], prior_point_2[2]-current_point_2[2]]
    prior_point_opp = [-1*prior_point_temp[0], -1*prior_point_temp[1], -1*prior_point_temp[2]]

    _, next_theta_base, next_phi_base = CartToSpherical(prior_point_opp[0], prior_point_opp[1], prior_point_opp[2])

    ## Naive approach, errors arise at/near the poles
    next_theta = np.random.normal(next_theta_base, arc_dist_size_rad_half/2)
    next_phi = np.random.normal(next_phi_base, arc_dist_size_rad_half/2)

    ## FIX ATTEMPT: use coordinate transformation matrix multiplication, ALMOST PASSES, troubles at the poles still
    new_base_theta = np.pi/2 # generate the distribution of points around the x axis, 
                                # which is at the equator in spherical coordinate system
    new_base_phi = 0
    next_theta_pre = np.random.normal(new_base_theta, arc_dist_size_rad_half/2)
    next_phi_pre = np.random.normal(new_base_phi, arc_dist_size_rad_half/2)

    next_x_pre, next_y_pre, next_z_pre = SphericalToCart(point_to_point, next_theta_pre, next_phi_pre)
    next_x, next_y, next_z = SphericalToCart(point_to_point, next_theta, next_phi)

    # Transform coordinate system so that norm distribution axis (x-axis) is now at the desired vector orientation
    vectorx = [next_x,next_y,next_z]
    uvx = vectorx / np.linalg.norm(vectorx)
    uvz = np.cross(uvx,[0,1,0])
    uvy = np.cross(uvx,uvz)

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


mlab.figure(bgcolor=(1,1,1),size=(800,800))

#mlab.points3d(bound_points_x, bound_points_y, bound_points_z, color = (0,0,0), scale_factor=1, opacity=1)

mlab.plot3d(points_x, points_y, points_z, color = (1,0,0), tube_radius = .05)
mlab.points3d(points_x, points_y, points_z, color = (0.2,0.2,0.2), scale_factor=0.2, opacity=1)
plotLead()
mlab.show()

def getLength(xs,ys,zs):
    l = 0
    for i in range(len(xs) - 1):
        dl = np.sqrt((xs[i + 1] - xs[i])**2 + (ys[i + 1] - ys[i])**2 + (zs[i + 1] - zs[i])**2)
        l = l + dl
    return l

#### Spline interpolate the tracts:
spline_xPosL = []
spline_yPosL = []
spline_zPosL = []

num_true_pts = 1000
u_fine = np.linspace(0,1,num_true_pts)

## Use a very precise (n = 1000) spline to determine the length of the fiber accurately
tck, u = splprep([points_x,points_y,points_z], s=spline_smoothing)
x_fine, y_fine, z_fine = splev(u_fine, tck)
tract_length = getLength(x_fine, y_fine, z_fine)
rounded_tract_length = round(tract_length * 2) / 2

## With this length, get the positions of the node compartments based on the internodal distance
num_step_points = int(rounded_tract_length / 0.5) + 1
u_step = np.linspace(0,1,num_step_points)
x_step, y_step, z_step = splev(u_step, tck)

# spline_xPosL.append(x_step)
# spline_yPosL.append(y_step)
# spline_zPosL.append(z_step)

mlab.figure(bgcolor=(1,1,1),size=(800,800))
mlab.plot3d(x_step, y_step, z_step, color = (0.2,0.2,0.2), tube_radius = .05)
plotLead()
mlab.show()