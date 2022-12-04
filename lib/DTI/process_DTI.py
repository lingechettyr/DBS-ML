'''
    This file enables the processing of diffusion tensor imaging (DTI) tractography based
    fiber tracts. From an input txt containing of 3d coordinate points, fiber
    tracts are individually resampled and spline interpolated to obtain realistic
    fiber trajectories. These tracts are then rediscretized to yield the center
    positions of each axon compartment along the fiber as specified by the MRG axon
    model. The coordinates along the center axis of the DBS lead are used to remove
    tracts that intersect with the lead or encapsulating scar tissue.
'''

import numpy as np
from scipy.interpolate import splprep
from scipy.interpolate import splev

class DTI_tracts:
    shift_fibers_to_origin = 0 # translate the DBS lead and surrounding pathways to the origin
    add_shift_fibers = 0 # translate the pathways independent of the DBS lead

    ## FIXME need to parameterize the below variables 
    take_every = 3 # Sample the DTI points at some interval as they are voxel-based, causing inaccurate jaggedness (can use '1' for artificial tracts, '3' for DTI tracts)
    spline_smoothing = .2 # smoothing constant chosen to minimize smoothing and retain small scale curvature
    node_count_requirement = 21 # Set a lower limit on the length of fibers to make predictions on
    
    #leftLeadPos = [[0,0],[0,0],[0,10]]
    leftLeadPos = [[167,161],[223,222],[143,159]] # use this line only if using anisotropic tissue conductivity
    tract_leftLeadPos = [[167,161],[223,222],[143,159]]

    if shift_fibers_to_origin == 1:
        leftLeadPos = [[0,0],[0,0],[0,10]]
        
    add_x_shift = -10
    add_y_shift = 20
    add_z_shift = 0

    def __init__(self, tracts_file, fem_bounds, node_to_node):
        self.xCompPos = []
        self.yCompPos = []
        self.zCompPos = []

        self.node_to_node = node_to_node

        #### Read in x,y,z positions from txt tract file, sample in desired interval
        with open(tracts_file) as file:
            for line in file:
                coords = [float(k) for k in line.split()]
                ## a single tract is stored on a single line in the following format:
                ## x1 y1 z1 x2 y2 z2 ... xn yn zn
                thisFiberX = coords[0::3]
                thisFiberY = coords[1::3]
                thisFiberZ = coords[2::3]

                ## Sample the positions of each tract to reduce noise/jaggedness
                self.xCompPos.append(thisFiberX[::self.take_every])
                self.yCompPos.append(thisFiberY[::self.take_every])
                self.zCompPos.append(thisFiberZ[::self.take_every])

        #### Spline interpolate the tracts:
        spline_xPosL = []
        spline_yPosL = []
        spline_zPosL = []

        num_true_pts = 1000
        u_fine = np.linspace(0,1,num_true_pts)
        for i in range(len(self.xCompPos)):

            ## Use a very precise (n = 1000) spline to determine the length of the fiber accurately
            tck, u = splprep([self.xCompPos[i],self.yCompPos[i],self.zCompPos[i]], s=self.spline_smoothing)
            x_fine, y_fine, z_fine = splev(u_fine, tck)
            tract_length = self.getLength(x_fine, y_fine, z_fine)
            rounded_tract_length = round(tract_length * 2) / 2

            ## With this length, get the positions of the node compartments based on the internodal distance
            num_step_points = int(rounded_tract_length / self.node_to_node) + 1
            u_step = np.linspace(0,1,num_step_points)
            x_step, y_step, z_step = splev(u_step, tck)

            spline_xPosL.append(x_step)
            spline_yPosL.append(y_step)
            spline_zPosL.append(z_step)

        if self.shift_fibers_to_origin == 1:
            spline_xPosL = np.subtract(spline_xPosL, self.tract_leftLeadPos[0][0])
            spline_yPosL = np.subtract(spline_yPosL, self.tract_leftLeadPos[1][0])
            spline_zPosL = np.subtract(spline_zPosL, self.tract_leftLeadPos[2][0])

        if self.add_shift_fibers == 1:
            spline_xPosL = np.subtract(spline_xPosL, self.add_x_shift)
            spline_yPosL = np.subtract(spline_yPosL, self.add_y_shift)
            spline_zPosL = np.subtract(spline_zPosL, self.add_z_shift)

        self.xCompPos = spline_xPosL
        self.yCompPos = spline_yPosL
        self.zCompPos = spline_zPosL

        #### Throw out tracts that pass through the DBS lead or surrounding 0.5 mm thick scar tissue
        dist_xPosL = []
        dist_yPosL = []
        dist_zPosL = []

        ## Set up interpolated line down center of DBS lead
        num_true_pts = 100
        tck, u = splprep([self.leftLeadPos[0],self.leftLeadPos[1],self.leftLeadPos[2]], s=.25, k=1)
        u_fine = np.linspace(0,1,num_true_pts)
        x_fine, y_fine, z_fine = splev(u_fine, tck)

        lead_tract_sep = 1.135 # 0.635 # Should be radius of lead OR scar tissue
        for i in range(len(self.xCompPos)): # For each tract in list of tracts
            remove = False
            for j in range(len(self.xCompPos[i])): # For each node in the tract
                for k in range(len(x_fine)): # Loop down center of DBS lead, make sure node is not within scar tissue
                    distance = np.sqrt((self.xCompPos[i][j] - x_fine[k])**2 + (self.yCompPos[i][j] - y_fine[k])**2 + (self.zCompPos[i][j] - z_fine[k])**2)
                    if distance <= lead_tract_sep:
                        remove = True
                        break

                if remove == True:
                    break

            if remove == False:
                dist_xPosL.append(self.xCompPos[i])
                dist_yPosL.append(self.yCompPos[i])
                dist_zPosL.append(self.zCompPos[i])

        self.xCompPos = dist_xPosL
        self.yCompPos = dist_yPosL
        self.zCompPos = dist_zPosL

        self.pre_trunc_xNodalComp = self.xCompPos
        self.pre_trunc_yNodalComp = self.yCompPos
        self.pre_trunc_zNodalComp = self.zCompPos

        #### Truncate tracts to fit within FEM bounds, ensure they meet node-count requirement
        trunc_xPosL = []
        trunc_yPosL = []
        trunc_zPosL = []

        for i in range(len(self.xCompPos)):
            tempFiberX = []
            tempFiberY = []
            tempFiberZ = []

            for j in range(len(self.xCompPos[i])): # For every nodal coordinate per tract, remove if not within FEM bounds
                if self.xCompPos[i][j] >= fem_bounds[0][0] and self.xCompPos[i][j] <= fem_bounds[0][1] and self.yCompPos[i][j] >= fem_bounds[1][0] and self.yCompPos[i][j] <= fem_bounds[1][1] and self.zCompPos[i][j] >= fem_bounds[2][0] and self.zCompPos[i][j] <= fem_bounds[2][1]:
                    tempFiberX.append(self.xCompPos[i][j])
                    tempFiberY.append(self.yCompPos[i][j])
                    tempFiberZ.append(self.zCompPos[i][j])

            if len(tempFiberX) >= self.node_count_requirement: # Ensure that the remaning tracts has the minimum number of nodes
                trunc_xPosL.append(tempFiberX)
                trunc_yPosL.append(tempFiberY)
                trunc_zPosL.append(tempFiberZ)

        self.xCompPos = trunc_xPosL
        self.yCompPos = trunc_yPosL
        self.zCompPos = trunc_zPosL

        self.xNodalComp = self.xCompPos
        self.yNodalComp = self.yCompPos
        self.zNodalComp = self.zCompPos

        print("Number of tracts that meet all critera: " + str(len(self.xNodalComp)))

    def getAllComps(self, lin_comp_pos):
        self.xAllComp = []
        self.yAllComp = []
        self.zAllComp = []

        #### Linearly interpolate between nodal points to get internodal compartment coordinates
        for i in range(len(self.xNodalComp)):
            xCompTemp = []
            yCompTemp = []
            zCompTemp = []
            for j in range(0,len(self.xNodalComp[i])-1):
                for k in range(len(lin_comp_pos)):
                    xCompTemp.append(self.xNodalComp[i][j] + ((self.xNodalComp[i][j+1] - self.xNodalComp[i][j]) * lin_comp_pos[k] / self.node_to_node))
                    yCompTemp.append(self.yNodalComp[i][j] + ((self.yNodalComp[i][j+1] - self.yNodalComp[i][j]) * lin_comp_pos[k] / self.node_to_node))
                    zCompTemp.append(self.zNodalComp[i][j] + ((self.zNodalComp[i][j+1] - self.zNodalComp[i][j]) * lin_comp_pos[k] / self.node_to_node))              
                    
            #add last nodal point 
            xCompTemp.append(self.xNodalComp[i][len(self.xNodalComp[i])-1])
            yCompTemp.append(self.yNodalComp[i][len(self.yNodalComp[i])-1])
            zCompTemp.append(self.zNodalComp[i][len(self.zNodalComp[i])-1])

            self.xAllComp.append(xCompTemp)
            self.yAllComp.append(yCompTemp)
            self.zAllComp.append(zCompTemp)


    def getNodeCompPos(self):
        return self.xNodalComp, self.yNodalComp, self.zNodalComp

    def getPreTruncNodeCompPos(self):
        return self.pre_trunc_xNodalComp, self.pre_trunc_yNodalComp, self.pre_trunc_zNodalComp

    def getAllCompPos(self):
        return self.xAllComp, self.yAllComp, self.zAllComp

    def getLength(self,xs,ys,zs):
        l = 0
        for i in range(len(xs) - 1):
            dl = np.sqrt((xs[i + 1] - xs[i])**2 + (ys[i + 1] - ys[i])**2 + (zs[i + 1] - zs[i])**2)
            l = l + dl
        return l

    def getTractCount(self):
        return int(len(self.xNodalComp))

    def getNodeCount(self, index):
        return int(len(self.xNodalComp[index]))

    def getLeadCoordinates(self):
        return self.leftLeadPos