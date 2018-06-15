# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 15:48:43 2018

@author: Henk
"""
import scipy as np
from scipy.special import comb
import pickle
#import state_estimation
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import MaxNLocator
import matplotlib as mpl

# global settings
plt.close('all')
bbox_props = dict(boxstyle="square,pad=0.9", alpha=1, fc="w", ec="k", lw=2)
np.random.seed(0)
do_plot = 0

# =============================================================================
# Load data from state estimation
# =============================================================================
with open('data.pickle', 'rb') as file:
    state_estimation = pickle.load(file)
    
X = state_estimation.z_pred
Y = state_estimation.c_m
T = state_estimation.T

mod_list = state_estimation.mod_list
mod_data = state_estimation.mod_data
mod_time = state_estimation.mod_time
mod_out  = state_estimation.mod_out

val_data = state_estimation.val_data
val_time = state_estimation.val_time

n  = 3 # model order
m  = np.size(X,0) # number of states
N  = np.size(X,1) # number of measurements
meas_list = range(N) # list of measurement indices
pm = comb(n+m, m, exact=True)

#plt.show(plt.plot(mod_time, mod_data[0], val_time, val_data[0]))

# modeling part
A = np.ones((len(mod_list), pm))

mod_idx = range(len(mod_time))

if n == 1: # first order model
    for i in mod_idx:
        for j in range(m):
            A[i,j+1] = mod_data[j,i]

elif n == 2: # second order model
    for i in mod_idx:
        for j in range(m):
            A[i,j+1  ] = mod_data[j,i]
            A[i,j+1+m] = mod_data[j,i]**2
        A[i,j+1+m+1] = mod_data[0,i]*mod_data[1,i]
        A[i,j+1+m+2] = mod_data[1,i]*mod_data[2,i]
        A[i,j+1+m+3] = mod_data[0,i]*mod_data[2,i]

elif n == 3: # third order model
    for i in mod_idx:
        for j in range(m):
            A[i,j+1      ] = mod_data[j,i]
            A[i,j+1+m    ] = mod_data[j,i]**2
            A[i,j+1+m+m+3] = mod_data[j,i]**3
        A[i,j+1+m+1] = mod_data[0,i]*mod_data[1,i]
        A[i,j+1+m+2] = mod_data[1,i]*mod_data[2,i]
        A[i,j+1+m+3] = mod_data[0,i]*mod_data[2,i]
        A[i,j+1+m+m+3+1] = mod_data[0,i]**2*mod_data[1,i]
        A[i,j+1+m+m+3+2] = mod_data[0,i]**2*mod_data[2,i]
        A[i,j+1+m+m+3+3] = mod_data[0,i]*mod_data[1,i]**2
        A[i,j+1+m+m+3+4] = mod_data[0,i]*mod_data[2,i]**2
        A[i,j+1+m+m+3+5] = mod_data[1,i]**2*mod_data[2,i]
        A[i,j+1+m+m+3+6] = mod_data[1,i]*mod_data[2,i]**2
        A[i,j+1+m+m+3+7] = mod_data[0,i]*mod_data[1,i]*mod_data[2,i]

elif n == 4: # fourth order model
    for i in mod_idx:
        for j in range(m):
            A[i,j+1          ] = mod_data[j,i]
            A[i,j+1+m        ] = mod_data[j,i]**2
            A[i,j+1+m+m+3    ] = mod_data[j,i]**3
            A[i,j+1+m+m+m+3+7] = mod_data[j,i]**4
        A[i,j+1+m+1] = mod_data[0,i]*mod_data[1,i]
        A[i,j+1+m+2] = mod_data[1,i]*mod_data[2,i]
        A[i,j+1+m+3] = mod_data[0,i]*mod_data[2,i]
        A[i,j+1+m+m+3+1] = mod_data[0,i]**2*mod_data[1,i]
        A[i,j+1+m+m+3+2] = mod_data[0,i]**2*mod_data[2,i]
        A[i,j+1+m+m+3+3] = mod_data[0,i]*mod_data[1,i]**2
        A[i,j+1+m+m+3+4] = mod_data[0,i]*mod_data[2,i]**2
        A[i,j+1+m+m+3+5] = mod_data[1,i]**2*mod_data[2,i]
        A[i,j+1+m+m+3+6] = mod_data[1,i]*mod_data[2,i]**2
        A[i,j+1+m+m+3+7] = mod_data[0,i]*mod_data[1,i]*mod_data[2,i]
        A[i,j+1+m+m+m+3+7+1] = mod_data[0,i]**3*mod_data[1,i]
        A[i,j+1+m+m+m+3+7+2] = mod_data[0,i]**3*mod_data[2,i]
        A[i,j+1+m+m+m+3+7+3] = mod_data[1,i]**3*mod_data[2,i]
        A[i,j+1+m+m+m+3+7+4] = mod_data[0,i]*mod_data[1,i]**3
        A[i,j+1+m+m+m+3+7+5] = mod_data[0,i]*mod_data[2,i]**3
        A[i,j+1+m+m+m+3+7+6] = mod_data[1,i]*mod_data[2,i]**3
        A[i,j+1+m+m+m+3+7+7] = mod_data[0,i]**2*mod_data[1,i]**2
        A[i,j+1+m+m+m+3+7+8] = mod_data[0,i]**2*mod_data[2,i]**2
        A[i,j+1+m+m+m+3+7+9] = mod_data[1,i]**2*mod_data[2,i]**2
        A[i,j+1+m+m+m+3+7+10] = mod_data[0,i]**2*mod_data[1,i]*mod_data[2,i]
        A[i,j+1+m+m+m+3+7+11] = mod_data[1,i]**2*mod_data[0,i]*mod_data[2,i]
        A[i,j+1+m+m+m+3+7+12] = mod_data[2,i]**2*mod_data[0,i]*mod_data[1,i]
        
else:
    print('Model order not implemented')
        
def plot_A(conf_arr):

    plt.close("all")
    conf_arr2 = np.zeros(np.shape(conf_arr))
    for i in range(len(conf_arr[:,0])):
        for j in range(len(conf_arr[0,:])):
            if conf_arr[i,j] != 1.:
                conf_arr2[i,j] = 1
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(np.array(conf_arr2), cmap='gray', aspect='auto',
                    interpolation='nearest')
    ax.grid()
    plt.show()

COV = np.linalg.pinv(A.T.dot(A))
VAR = np.diag(COV)
Chat = COV.dot(A.T)
Chat = Chat.dot(mod_out)
stuff = A.dot(Chat)
#plt.show(plt.plot(stuff[:]))

# restore to full resolution using interpolation
mod_rest = np.interpolate.griddata(mod_time, stuff,    T)
val_rest = np.interpolate.griddata(val_time, val_data, T)
#val_rest = [val_rest[i+1] if np.isnan(val_rest[i]) else val_rest[i] for i in range(len(val_rest))]
val_rest[0] = val_rest[1]
plt.plot(mod_rest, 'b-+')
plt.plot(val_rest, 'r-+')
plt.grid()
plt.show()

residual = mod_rest-val_rest
print(residual)
correlation = np.correlate(residual-np.mean(residual), residual-np.mean(residual), mode = 2)
n_lags   = len(correlation)
lags     = np.arange(-n_lags/2, n_lags/2)+1

start = 0
end   = -1

#==============================================================================
# Plotting
#==============================================================================
do_plot = 1
if do_plot:
    
    plt.close("all")
    #plt.plot(XX_k1k1[0,:]-X_k[0,:])
    
    fig = plt.figure()
    
    #ax1 = fig.add_subplot(221)
    ##ax1.title.set_text(r'something')
    #ax1.plot(np.transpose(Y[0,start:end]), 'g', label=r'$\alpha_{post\,state-estimation}$')
    #ax1.plot(stuff[start:end,0], color='r', linewidth=1, label=r'$\alpha_{post\,LS}$')
    #ax1.set_xlabel(r'$Time\,[ms]$')
    #ax1.set_ylabel(r'$\alpha\,[rad]$')
    #ax1.legend(shadow=True)
    
    ax1 = fig.add_subplot(221)
    #ax1.title.set_text(r'something')
    #ax1.plot(Y[0,start:end], 'g', label=r'$\alpha_{post\,state-estimation}$')
    ax1.plot(T[start:end], Y[start:end], label=r'$Measured \, C_m$')
    ax1.plot(T[start:end], mod_rest[start:end], color='r', linewidth=1, label=r'$Modelled \, C_m$')
    ax1.grid()
    ax1.set_xlabel(r'$Time \, [ms]$')
    ax1.set_ylabel(r'$C_m \, [-]$')
    ax1.legend(shadow=True)
    
    ax1 = fig.add_subplot(222)
    ax1.plot(residual[start:end], color='b', linewidth=1, label=r'$Residual \, C_m$')
    mean = np.mean(residual)
    ax1.plot([mean]*len(residual), color='r', linewidth=1,
             label=r'$Mean \, Residual \, C_m = \, {:5.4f}$'.format(mean))
    ax1.grid()
    string = r'$Sum \, of \, squared \, residuals \, = \, {:5.4f}$'.format(
             np.sum(np.square(residual)))
    
    ax1.text(0.9, 0.1, string, ha='right', va='center', transform=ax1.transAxes,
             bbox = bbox_props)
    ax1.set_xlabel(r'$Time \, [ms]$')
    ax1.set_ylabel(r'$\Delta C_m \, [-]$')
    ax1.legend(shadow=True)
    
    ax = fig.add_subplot(223)
    #ax.acorr(residual[start:end]-np.mean(residual), maxlags = None,
    #    usevlines = False, label=r'$Autocorrelation \, Residual$')[2]
    ax.plot(lags, correlation, color='b', label=r'$Correlation \, function$')
    ax.plot(lags,  [1.96 / np.sqrt(len(residual))]*n_lags, 'r--',
                    label=r'$95\% \, Confidence \, bounds$')
    ax.plot(lags, [-1.96 / np.sqrt(len(residual))]*n_lags, 'r--')
    ax.grid()
    ax.set_xlabel(r'$Number \, of \, lags$')
    ax.set_ylabel(r'$C_m\,[-]$')
    ax.legend(shadow=True)
    
    #ax6 = fig.add_subplot(222)
    ##ax6.title.set_text(r'$somteing$')
    #ax6.plot(np.transpose(Y[1,start:end]), 'g', label=r'$\beta_{post\,state-estimation}$')
    #ax6.plot(stuff[start:end,1], color='r', linewidth=1, label=r'$\beta_{post\,LS}$')
    #ax6.set_xlabel(r'$Time\,[ms]$')
    #ax6.set_ylabel(r'$\beta\,[rad]$')
    #ax6.legend(shadow=True)
    #
    #ax7 = fig.add_subplot(223)
    ##ax7.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    #ax7.plot(np.transpose(Y[2,start:end]), 'g', label=r'$V_{post\,state-estimation}$')
    #ax7.plot(stuff[start:end,2], color='r', linewidth=1, label=r'$V_{post\,LS}$')
    #ax7.set_xlabel(r'$Time\,[ms]$')
    #ax7.set_ylabel(r'$V_{tot}\,[m/s]$')
    #ax7.legend(shadow=True)
    
    #ax7 = fig.add_subplot(234)
    #ax7.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    #ax7.plot(np.transpose(U[0,start:end]), 'g', label=r'$V_{tot}$')
    #ax7.plot(stuff[start:end,2], color='r', linewidth=2, label=r'$V_{tot}$')
    #ax7.set_xlabel(r'$Time\,[s]$')
    #ax7.set_ylabel(r'$V_{tot}\,[m/s]$')
    #ax7.legend(shadow=True)
    
    #ax7 = fig.add_subplot(235)
    #ax7.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    #ax7.plot(np.transpose(U[1,start:end]), 'g', label=r'$V_{tot}$')
    #ax7.plot(stuff[start:end,2], color='r', linewidth=2, label=r'$V_{tot}$')
    #ax7.set_xlabel(r'$Time\,[s]$')
    #ax7.set_ylabel(r'$V_{tot}\,[m/s]$')
    #ax7.legend(shadow=True)
    
    #ax7 = fig.add_subplot(236)
    #ax7.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    #ax7.plot(np.transpose(U[2,start:end]), 'g', label=r'$V_{tot}$')
    #ax7.plot(stuff[start:end,2], color='r', linewidth=2, label=r'$V_{tot}$')
    #ax7.set_xlabel(r'$Time\,[s]$')
    #ax7.set_ylabel(r'$V_{tot}\,[m/s]$')
    #ax7.legend(shadow=True)
    
    ax = fig.add_subplot(224)
    ax.grid()
    ax.title.set_text(r'$Variance\,per\,coefficient$')
    n_coef = np.arange(len(VAR))
    ax.bar(n_coef, VAR, align='center', log=True, color='g', label=r'$Variance$')
    ax.set_xlabel(r'$Coefficient$')
    ax.set_ylabel(r'$VAR$')
    ax.set_xlim([min(n_coef)-0.5, max(n_coef)+0.5])
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.legend(shadow=True)
    
    plt.show()
    
    TRI  = mpl.tri.Triangulation(mod_data[0], mod_data[1])
    mask = mpl.tri.TriAnalyzer(TRI).get_flat_tri_mask(0.01)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(mod_data[0], mod_data[1], stuff, mask=mask, cmap=cm.jet, linewidth=0.2)
    ax.set_xlabel(r'$\alpha$')
    ax.set_ylabel(r'$\beta$')
    ax.set_zlabel(r'$C_m$')
    
    plt.show()
    
    #plot_A(A)