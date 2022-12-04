'''
    The script graphs the fiber tracts orthogonal to the DBS lead.
'''

import numpy as np
from mayavi import mlab
from scipy.interpolate import splprep
from scipy.interpolate import splev
import sys


leftLeadPos = [[0,0],[0,0],[0,10]]
node_to_node = 0.5

lines = int(sys.argv[1])

add_x_shift = -10
add_y_shift = 20
add_z_shift = 0

leadheight = 20
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
    position = (1.5 * 7)
    height_step = np.arctan2(leadheight,radius)
    phi, theta = np.meshgrid(np.arange(np.pi/2,np.pi/2 + height_step+height_step,height_step), np.arange(0,2*np.pi+step,step))
    x = np.sin(phi) * np.cos(theta) * radius
    y = np.sin(phi) * np.sin(theta) * radius
    z = np.cos(phi) * radius
    mult = 1 / np.cos(height_step)
    x[:,1] *= mult
    y[:,1] *= mult
    z[:,1] *= mult
    z = np.add(z,leadheight+position)
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

xCompPos = []
yCompPos = []
zCompPos = []

lead_radius = (1.27/2)
scar_radius = 0.5
contact_vertical_center = 6.75

x = np.arange(-10, 10, node_to_node)
y = np.arange(lead_radius+scar_radius, 12.5+lead_radius+0.1, 0.8)
z = np.arange(contact_vertical_center-12, contact_vertical_center+12+0.1, 0.8)

if lines == 1:
    for y_step in y:
        for z_step in z:
            temp_x = []
            temp_y = []
            temp_z = []
            for x_step in x:
                temp_x.append(x_step)
                temp_y.append(y_step)
                temp_z.append(z_step)
            xCompPos.append(temp_x)
            yCompPos.append(temp_y)
            zCompPos.append(temp_z)
else:
    for y_step in y:
        for z_step in z:
            xCompPos.append(0)
            yCompPos.append(y_step)
            zCompPos.append(z_step)

mlab.figure(bgcolor=(1,1,1),size=(800,800))

colors = [(0.5,0.5,0.5)]

for tract_index in range(len(xCompPos)):
    if lines == 1:
        mlab.plot3d(xCompPos[tract_index], yCompPos[tract_index],zCompPos[tract_index], color = colors[0], tube_radius = .05)
    else:
        mlab.points3d(xCompPos[tract_index], yCompPos[tract_index],zCompPos[tract_index], color = (0,0,0), scale_factor=.2, opacity=1)

#mlab.points3d([0], [0], [6.75], color = (0,0,1), scale_factor=3, opacity=1)

plotLead()
mlab.show()