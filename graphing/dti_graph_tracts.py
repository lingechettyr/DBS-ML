'''
    The script graphs the fiber pathways from a specified file. The user can choose whether to
    spline interpolate the tracts as well as whether to shift them to the origin. This script is
    mainly used to test different transformation techniques before adding them to process_DTI.py.
'''

import numpy as np
from mayavi import mlab
from scipy.interpolate import splprep
from scipy.interpolate import splev
import sys

tracts_file = sys.argv[1]
tracts_per_pathway = int(sys.argv[2])

shift_fibers_to_origin = 0 # translate the DBS lead and surrounding pathways to the origin
add_shift_fibers = 0 # translate the pathways independent of the DBS lead
removeScar = 0 # remove any fibers that intersect with the DBS lead or scar tissue
truncateToBounds = 0 # truncate the fibers to lie within certain boundaries
bound_center = [0,0,6.5]
bound_distance = 10
bounds = [[bound_center[0]-bound_distance, bound_center[0]+bound_distance],
          [bound_center[1]-bound_distance, bound_center[1]+bound_distance],
          [bound_center[2]-bound_distance, bound_center[2]+bound_distance]]

node_count_requirement = 21

spline = 0
take_every = 3
spline_smoothing = .2

rotate = 0

#leftLeadPos = [[0,0],[0,0],[0,10]]
leftLeadPos = [[167,161],[223,222],[143,159]]
tract_leftLeadPos = [[167,161],[223,222],[143,159]]
node_to_node = 0.5

if shift_fibers_to_origin == 1:
    leftLeadPos = [[0,0],[0,0],[0,10]]

add_x_shift = -10
add_y_shift = 20
add_z_shift = 0

leadheight = 40
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

def getLength(xs,ys,zs):
    l = 0
    for i in range(len(xs) - 1): # FIXME does squared need paranethesis??
        dl = np.sqrt((xs[i + 1] - xs[i])**2 + (ys[i + 1] - ys[i])**2 + (zs[i + 1] - zs[i])**2)
        l = l + dl
    return l

xCompPos = []
yCompPos = []
zCompPos = []

take_every_line = 1
itr = 0

#### Read in x,y,z positions from txt tract file, sample in desired interval
with open(tracts_file) as file:
    for line in file:
        if itr % take_every_line != 0:
            itr += 1
            continue

        coords = [float(k) for k in line.split()]
        ## a single tract is stored on a single line in the following format:
        ## x1 y1 z1 x2 y2 z2 ... xn yn zn
        thisFiberX = coords[0::3]
        thisFiberY = coords[1::3]
        thisFiberZ = coords[2::3]

        ## Sample the positions of each tract to reduce noise/jaggedness
        xCompPos.append(thisFiberX[::take_every])
        yCompPos.append(thisFiberY[::take_every])
        zCompPos.append(thisFiberZ[::take_every])

        itr += 1

number_of_pathways = int(len(xCompPos) / tracts_per_pathway)

if spline == 1:
    #### Spline interpolate the tracts:
    spline_xPosL = []
    spline_yPosL = []
    spline_zPosL = []

    num_true_pts = 1000
    u_fine = np.linspace(0,1,num_true_pts)
    for i in range(len(xCompPos)):

        ## Use a very precise (n = 1000) spline to determine the length of the fiber accurately
        tck, u = splprep([xCompPos[i],yCompPos[i],zCompPos[i]], s=spline_smoothing)
        x_fine, y_fine, z_fine = splev(u_fine, tck)
        tract_length = getLength(x_fine, y_fine, z_fine)
        rounded_tract_length = round(tract_length * 2) / 2

        ## With this length, get the positions of the node compartments based on the internodal distance
        num_step_points = int(rounded_tract_length / node_to_node) + 1
        u_step = np.linspace(0,1,num_step_points)
        x_step, y_step, z_step = splev(u_step, tck)

        spline_xPosL.append(x_step)
        spline_yPosL.append(y_step)
        spline_zPosL.append(z_step)

    xCompPos = spline_xPosL
    yCompPos = spline_yPosL
    zCompPos = spline_zPosL

pathway_remove_counts = [0 for k in range(number_of_pathways)] #np.zeros(number_of_pathways)

if removeScar == 1:
    #### Throw out tracts that pass through the DBS lead or surrounding 0.5 mm thick scar tissue
    dist_xPosL = []
    dist_yPosL = []
    dist_zPosL = []

    ## Set up interpolated line down center of DBS lead
    num_true_pts = 100
    tck, u = splprep([leftLeadPos[0],leftLeadPos[1],leftLeadPos[2]], s=.25, k=1)
    u_fine = np.linspace(0,1,num_true_pts)
    x_fine, y_fine, z_fine = splev(u_fine, tck)

    lead_tract_sep = 1.135 # 0.635 # Should be radius of lead OR scar tissue
    for i in range(len(xCompPos)): # For each tract in list of tracts
        remove = False
        for j in range(len(xCompPos[i])): # For each node in the tract
            for k in range(len(x_fine)): # Loop down center of DBS lead, make sure node is not within scar tissue
                distance = np.sqrt((xCompPos[i][j] - x_fine[k])**2 + (yCompPos[i][j] - y_fine[k])**2 + (zCompPos[i][j] - z_fine[k])**2)
                if distance <= lead_tract_sep:
                    remove = True
                    break

            if remove == True:
                #print(i)
                #print(tracts_per_pathway)
                #print(len(pathway_remove_counts))
                pathway_remove_counts[int((i)/tracts_per_pathway)] += 1
                break

        if remove == False:
            dist_xPosL.append(xCompPos[i])
            dist_yPosL.append(yCompPos[i])
            dist_zPosL.append(zCompPos[i])

    xCompPos = dist_xPosL
    yCompPos = dist_yPosL
    zCompPos = dist_zPosL

print(pathway_remove_counts)
tracts_in_pathways = [tracts_per_pathway - pathway_remove_counts[k] for k in range(len(pathway_remove_counts))]
print(tracts_in_pathways)

if shift_fibers_to_origin == 1:
    xCompPos = np.subtract(xCompPos, tract_leftLeadPos[0][0])
    yCompPos = np.subtract(yCompPos, tract_leftLeadPos[1][0])
    zCompPos = np.subtract(zCompPos, tract_leftLeadPos[2][0])

## This function does not work yet, FIXME
if rotate == 1:

    # Get original DBS lead coordinates
    x_temp = leftLeadPos[0] #[119, 122]
    y_temp = leftLeadPos[1] #[78, 78]
    z_temp = leftLeadPos[2] #[64, 72]

    # # First translate original dbs lead position to origin
    x_trans = np.subtract(x_temp,x_temp[0])
    y_trans = np.subtract(y_temp,y_temp[0])
    z_trans = np.subtract(z_temp,z_temp[0])

    # First translate original dbs lead position to origin
    #x_trans = [0,0]
    #y_trans = [0,-10]
    #z_trans = [0,-10]

    # "uva = unit vector a"
    vectorz = [x_trans[1],y_trans[1],z_trans[1]]
    uvz = vectorz / np.linalg.norm(vectorz)
    uvx = np.cross(uvz,[0,1,0])
    uvy = np.cross(uvz,uvx)

    # rotation_matrix = np.matrix([[uvx[0],uvx[1],uvx[2],0],
    #                             [uvy[0],uvy[1],uvy[2],0],
    #                             [uvz[0],uvz[1],uvz[2],0],
    #                             [0     ,0     ,0     ,1]])

    rotation_matrix = np.matrix([[uvx[0],uvy[0],uvz[0],0],
                                [uvx[1],uvy[1],uvz[1],0],
                                [uvx[2],uvy[2],uvz[2],0],
                                [0     ,0     ,0     ,1]])

    for i in range(len(spline_xPosL)):
        spline_xPosL_trans = np.subtract(spline_xPosL[i], x_temp[0])
        spline_yPosL_trans = np.subtract(spline_yPosL[i], y_temp[0])
        spline_zPosL_trans = np.subtract(spline_zPosL[i], z_temp[0])

        rotx = []
        roty = []
        rotz = []

        for j in range(len(spline_xPosL_trans)):
            point = [spline_xPosL_trans[j], spline_yPosL_trans[j], spline_zPosL_trans[j], 1]
            rot_point = np.matmul(rotation_matrix, point)
            rotx.append(rot_point[0,0])
            roty.append(rot_point[0,1])
            rotz.append(rot_point[0,2])

        spline_xPosL[i] = np.add(rotx, x_temp[0])
        spline_yPosL[i] = np.add(roty, y_temp[0])
        spline_zPosL[i] = np.add(rotz, z_temp[0])

if add_shift_fibers == 1:
    xCompPos = np.subtract(xCompPos, add_x_shift)
    yCompPos = np.subtract(yCompPos, add_y_shift)
    zCompPos = np.subtract(zCompPos, add_z_shift)

if truncateToBounds == 1:
    #### Truncate tracts to fit within FEM bounds, ensure they meet node-count requirement
    trunc_xPosL = []
    trunc_yPosL = []
    trunc_zPosL = []

    for i in range(len(xCompPos)):
        tempFiberX = []
        tempFiberY = []
        tempFiberZ = []

        for j in range(len(xCompPos[i])): # For every nodal coordinate per tract, remove if not within FEM bounds
            if xCompPos[i][j] >= bounds[0][0] and xCompPos[i][j] <= bounds[0][1] and yCompPos[i][j] >= bounds[1][0] and yCompPos[i][j] <= bounds[1][1] and zCompPos[i][j] >= bounds[2][0] and zCompPos[i][j] <= bounds[2][1]:
                tempFiberX.append(xCompPos[i][j])
                tempFiberY.append(yCompPos[i][j])
                tempFiberZ.append(zCompPos[i][j])

        if len(tempFiberX) >= node_count_requirement: # Ensure that the remaning tracts has the minimum number of nodes
                trunc_xPosL.append(tempFiberX)
                trunc_yPosL.append(tempFiberY)
                trunc_zPosL.append(tempFiberZ)

    xCompPos = trunc_xPosL
    yCompPos = trunc_yPosL
    zCompPos = trunc_zPosL

mlab.figure(bgcolor=(1,1,1),size=(800,800))

colors = [(0.5,0.5,0.5), (1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
# colors = [(.9,0,1), (0,1,0), (0,0,1), (0,0,1), (1,1,0), (1,0,1), (0,1,1)] #(1,0.5,0),
#colors = [(0,1,0), (0,0,1), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
#colors = [(0,0,1), (0,0,1), (1,1,0), (1,0,1), (0,1,1)]
#colors = [(0.5,0.5,0.5)]


if truncateToBounds == 1:
    for j in range(len(xCompPos)):
        tract_index = j
        mlab.plot3d(xCompPos[tract_index], yCompPos[tract_index],zCompPos[tract_index], color = colors[0], tube_radius = .05)
else:
    base_tract_index = 0
    for i in range(number_of_pathways):
        for j in range(tracts_in_pathways[i]):
            tract_index = j + base_tract_index
            mlab.plot3d(xCompPos[tract_index], yCompPos[tract_index],zCompPos[tract_index], color = colors[i], tube_radius = .05)
        base_tract_index += tracts_in_pathways[i]

plotLead()
mlab.show()