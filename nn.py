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
from simnet_old import simnet
from plot_struct import plot_struct
from bar import bar_init, bar


import matplotlib.pyplot as plt
#from matplotlib import cm
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
#from matplotlib.colors import LightSource



# global settings
plt.close('all')
bbox_props = dict(boxstyle="square,pad=0.9", alpha=1, fc="w", ec="k", lw=2)
#np.random.seed(0)
#random_state = np.random.get_state()
#np.random.set_state(random_state)

# Load data from state estimation step

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


elif netRBF.NhidL == 2:
    Nin = 2
    res = 2
    center_distr = np.zeros((Nin, res))
    netRBF.centers1 = np.zeros((Nin, res**Nin))
    for i in range(Nin):
        center_distr[i] = np.linspace(min(mod_data[i]), max(mod_data[i]), res)
    mesh = np.meshgrid(*center_distr)

    for i in range(Nin):
        centerpos_dim_i = np.reshape(mesh[i], (np.size(mesh[i])))
        netRBF.centers1[i] = centerpos_dim_i
#    print(netRBF.centers)
        
    Nhid2 = np.size(mesh[i])
    center_distr = np.zeros((Nhid2, res))
    netRBF.centers2 = np.zeros((Nin, res**Nin))
    for i in range(Nhid2):
        center_distr[i,:] = np.linspace(min(mod_data[i]), max(mod_data[i]), res)
    mesh = np.meshgrid(*center_distr)

    for i in range(Nin):
        centerpos_dim_i = np.reshape(mesh[i], (np.size(mesh[i])))
        netRBF.centers2[i] = centerpos_dim_i
    
    netRBF.IW = np.random.uniform(0, 1, (netRBF.Nin, netRBF.Nhid2))
    netRBF.HW = np.random.uniform(0, 1, (netRBF.Nhid2, netRBF.Nhid))
    netRBF.OW = np.random.uniform(-0.1, 0.1, (netRBF.Nhid, netRBF.Nout))*-1

else:
    print("This layer amount is not implemented. Please choose between {}.".format("1 and 2"))

#plt.show(plt.scatter(centersx, centersy)) # plot of the locations of the RBFs
# =============================================================================
# Try X times, then rerun with the best result, and show that
# =============================================================================

hm_try = 10 #how many tries, with different initial conditions
parameters = {}

#bar_init()
for i in range(hm_try):
    random_state = np.random.get_state()
    netRBF.IW = np.random.uniform(0.9, 1.1, (netRBF.NinB, netRBF.NhidB))
    netRBF.OW = np.random.uniform(-4, 4, netRBF.NhidB)
    yRBF = simnet(netRBF, Xeval, mod_out, verboose=True)
    parameters[i] = [yRBF[2], random_state]
#    bar(i, hm_try)
    
print('\n')


results = [parameters[i][0] for i in range(len(parameters))]
plt.plot(results)
plt.grid()
plt.show()
best_result = results.index(min(results))
print('Best result on try', '{},'.format(best_result), 'Best Final RMS:', '{}'.format(results[best_result]))

np.random.set_state(parameters[best_result][1])
netRBF.IW = np.random.uniform(0.9, 1.1, (netRBF.NinB, netRBF.NhidB))
netRBF.OW = np.random.uniform(-4, 4, netRBF.NhidB)
yRBF = simnet(netRBF, Xeval, mod_out, verboose=False)

#==============================================================================
# Plotting
#==============================================================================

#TRIeval = np.spatial.Delaunay(Xeval.T)#, qhull_options = 'Qbb Qc QJ1e-6')


do_plot = 1
animate = 0 #enable this to get a rotating animation

if do_plot:
    TRI  = mpl.tri.Triangulation(Xeval[0], Xeval[1])
    mask = mpl.tri.TriAnalyzer(TRI).get_flat_tri_mask(0.01)
    
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90,0)
#    ax.set_proj_type('ortho')
    ax.plot_trisurf(Xeval[0], Xeval[1], yRBF[0], cmap=mpl.cm.jet, linewidth=0.2)
#    ax.scatter(X[0], X[1], Y, c='k', marker='.', linewidths=0.1)
    ax.scatter(centersx, centersy, -0.06, c='k') # plots the RBF centers
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

#    plot_struct(netRBF)
#if __name__ is '__main__':
#    main()
