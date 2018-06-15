# -*- coding: utf-8 -*-
"""
Created on Tue Apr 24 17:33:48 2018

@author: Henk
"""

import scipy as np
import scipy.linalg as la
import pickle
from rbf_data import netRBF
#from simnet import simnet
#from simnet_old import simnet
#from plot_struct import plot_struct

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import LightSource

#np.set_printoptions(precision=2)

plt.close('all')
bbox_props = dict(boxstyle="square,pad=0.9", alpha=1, fc="w", ec="k", lw=2)

#np.random.seed(0)
#random_state = np.random.get_state()
#np.random.set_state(random_state)

# =============================================================================
# Load data from state estimation
# =============================================================================

with open('data.pickle', 'rb') as file:
    state_estimation = pickle.load(file)

X = state_estimation['z_pred']
Y = state_estimation['c_m']
T = state_estimation['T']

mod_list = state_estimation['mod_list']
mod_data = state_estimation['mod_data']
mod_time = state_estimation['mod_time']
mod_out  = state_estimation['mod_out']

val_data = state_estimation['val_data']
val_time = state_estimation['val_time']


m  = np.size(X,0) # number of states
N  = np.size(X,1) # number of measurements
meas_list = range(N) # list of measurement indices

x1 = mod_data[0]
x2 = mod_data[1]

bias = netRBF.biases
biases = np.ones((1,len(x1)))
Xeval = np.row_stack((x1, x2, biases)) if bias else np.row_stack((x1, x2))


#netRBF.NhidL = 1
if netRBF.NhidL == 1:
    if not netRBF.distr:
        center_x = np.linspace(min(x1), max(x1), 4)
        center_y = np.linspace(min(x2), max(x2), 4)
        centersx, centersy = np.meshgrid(center_x, center_y)
        size_centers = np.size(centersx)
        centersx = np.reshape(centersx, (np.size(centersx)))
        centersy = np.reshape(centersy, (np.size(centersy)))
        
        netRBF.centers = np.zeros((3, size_centers)) if bias else np.zeros((2, size_centers))

        netRBF.Nhid = size_centers
        netRBF.NhidB = (size_centers+1) if bias else size_centers
    
        netRBF.centers[0] = centersx
        netRBF.centers[1] = centersy
        if bias:
            netRBF.centers[2] = np.ones((size_centers)) 

    elif netRBF.distr == 'smart':
        netRBF.centers[0] = centersx
        netRBF.centers[1] = centersy

else:
    print("This layer amount is not implemented. Please choose between {}.".format("1 and 2"))

#plt.show(plt.scatter(centersx, centersy)) # plot of the locations of the RBFs
# =============================================================================
# Try X times, then rerun with the best result, and show that
# =============================================================================

netRBF.IW = np.ones((netRBF.NinB, netRBF.NhidB))
netRBF.OW = np.ones((netRBF.NhidB, netRBF.Nout))

A = np.zeros((len(mod_out),size_centers))

#i = number of measurements
#j = number of neurons
#k = number of input dimensions

for j in range(size_centers):
    for i in range(len(mod_out)):
        x_i = mod_data[0:2,[i]]
        nu_ij = la.norm((netRBF.IW[:,[j]])**2 * (x_i - netRBF.centers[:,[j]])**2)
        A[i,j] = np.exp(-nu_ij)

netRBF.OW = np.linalg.inv(A.T.dot(A)).dot(A.T).dot(mod_out)

Output = A.dot(netRBF.OW)

print("RMS: {:6f}".format(np.sqrt(np.sum((Output - mod_out)**2)/len(mod_out))))

#==============================================================================
# Plotting
#==============================================================================

do_plot = 1
animate = 0 #enable this to get a rotating animation

if do_plot:
    TRI  = mpl.tri.Triangulation(Xeval[0], Xeval[1])
    mask = mpl.tri.TriAnalyzer(TRI).get_flat_tri_mask(0.01)
    
    fig  = plt.figure()
    ax   = fig.add_subplot(111, projection='3d')
    ax.view_init(90,0)
#    ax.set_proj_type('ortho')
    ax.plot_trisurf(Xeval[0], Xeval[1], Output, mask=mask, cmap=mpl.cm.jet, linewidth=0.2)
#    ax.scatter(X[0], X[1], Y, c='k', marker='.', linewidths=0.1)
#    ax.scatter(centersx, centersy, -0.06, c='k') # plots the RBF centers
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$C_m$')

    if animate:
        for angle in range(0, 360):
            ax.view_init(30, angle)
            plt.draw()
            plt.pause(.001)
    else:
        plt.show()
