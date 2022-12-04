'''
    This file provides functions for graphing the DTI-based fiber tracts
    and DBS lead, using the Mayavi library.
'''

import numpy as np
from mayavi import mlab

class DTI_grapher:
    
    def __init__(self, leftLeadPos, xNodalComp, yNodalComp, zNodalComp):
        self.leftLeadPos = leftLeadPos
        self.xNodalComp = xNodalComp
        self.yNodalComp = yNodalComp
        self.zNodalComp = zNodalComp
        
    def plotTracts(self):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))

        for i in range(len(self.xNodalComp)):
            mlab.plot3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], color = (0,1,0), tube_radius = .05)
            #mlab.points3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], np.full(len(self.xNodalComp[i]),.25), color = (0,0,1), scale_factor=1)

        self.plotLead()
        mlab.show()

    def plotActivatedTracts(self, act_inds, prob_inds):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))

        for i in range(len(self.xNodalComp)):
            if i in prob_inds:
                continue
            if i in act_inds:
                mlab.plot3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], color = (1,0,0), tube_radius = .05)
            else:
                mlab.plot3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], color = (0.7,0.7,0.7), tube_radius = .05)

        self.plotLead()
        mlab.show()

    def plotSingleTract(self, ind):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))

        mlab.plot3d(self.xNodalComp[ind], self.yNodalComp[ind],self.zNodalComp[ind], color = (0.7,0.7,0.7), tube_radius = .05)

        self.plotLead()
        mlab.show()

    def plotSingleTractColorMap(self, ind, voltages):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))
        voltages = np.reshape(voltages, (len(voltages)))

        #mlab.points3d(np.asarray(self.xNodalComp[ind]), np.asarray(self.yNodalComp[ind]), np.asarray(self.zNodalComp[ind]), np.asarray(voltages), colormap = "blue-red", scale_factor=0.30, scale_mode="none", opacity=1)
        # Below to plot as continuous line
        mlab.plot3d(np.asarray(self.xNodalComp[ind]), np.asarray(self.yNodalComp[ind]), np.asarray(self.zNodalComp[ind]), np.asarray(voltages), colormap = "blue-red", tube_radius = .05, vmin=0, vmax=150)

        self.plotLead()
        mlab.show()

    def plotTractsColorMap(self, voltages, max_voltage):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))
        for i in range(len(self.xNodalComp)):
            voltages_ind = np.reshape(voltages[i], (len(voltages[i])))
            mlab.plot3d(np.asarray(self.xNodalComp[i]), np.asarray(self.yNodalComp[i]), np.asarray(self.zNodalComp[i]), np.asarray(voltages_ind), colormap = "blue-red", tube_radius = .05, vmin=0, vmax=max_voltage)

        self.plotLead()
        mlab.show()

    def plotTractsColorMap_temp(self, voltages, max_voltage, i):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))
        voltages_ind = np.reshape(voltages[i], (len(voltages[i])))
        mlab.plot3d(np.asarray(self.xNodalComp[i]), np.asarray(self.yNodalComp[i]), np.asarray(self.zNodalComp[i]), np.asarray(voltages_ind), colormap = "blue-red", tube_radius = .05, vmin=0, vmax=max_voltage)

        self.plotLead()
        mlab.show()

    def plotActivatedTracts_temp(self, act_inds, prob_inds, i):
        mlab.figure(bgcolor=(1,1,1),size=(800,800))

        if i in act_inds:
            mlab.plot3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], color = (1,0,0), tube_radius = .05)
        else:
            mlab.plot3d(self.xNodalComp[i], self.yNodalComp[i],self.zNodalComp[i], color = (0.7,0.7,0.7), tube_radius = .05)

        self.plotLead()
        mlab.show()

    # Plotting Code #
    def plotLead(self):
        ## Set up rotation ##
        lead_x = [0,0]
        lead_y = [0,0]
        lead_z = [0,10]

        # Get original DBS lead coordinates
        x_temp = self.leftLeadPos[0] #[119, 122]
        y_temp = self.leftLeadPos[1] #[78, 78]
        z_temp = self.leftLeadPos[2] #[64, 72]

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
        height = 100
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
