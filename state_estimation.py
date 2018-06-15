# -*- coding: utf-8 -*-
"""
Created on Fri Apr 06 11:29:40 2018

@author: Henk
"""

import scipy as np
#import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from matplotlib import cm
#from mpl_toolkits.mplot3d import Axes3D
#import control.matlab as control

plt.close('all')
plt.rc('text', usetex=False)

mat = loadmat('../F16traindata_CMabV_2018.mat')
z_k = np.transpose(mat['Z_k'])
c_m = mat['Cm'][:,0]
u_k = np.transpose(mat['U_k'])

alpha_m = z_k[0,:]
beta_m  = z_k[1,:]
Vtot    = z_k[2,:]

Au = u_k[0,:]
Av = u_k[1,:]
Aw = u_k[2,:]

def plot_data():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(90,0)
    ax.plot_trisurf(alpha_m, beta_m, c_m, cmap=cm.jet)
    ax.set_xlabel(r'$\alpha_m$')
    ax.set_ylabel(r'$\beta_m$')
    ax.set_zlabel(r'$V_{tot}$')
    plt.show()
    
def kf_calcFx(t, x, u):
    """system dynamics equation f(x,u,t)"""
    n = np.size(x)
    xdot = np.zeros((n))
    if n == 4:
        xdot[0] = u[0]
        xdot[1] = u[1]
        xdot[2] = u[2]
        xdot[3] = 0.
    else:
        print('Alarm')
#    print('xdot=', xdot)
    return xdot
    
    
def kf_calcDFx(t, x, u):
    """Calculate Jacobian matrix of f(x,u,t)"""
    n = np.size(x, 0)
    DFx = np.zeros((n, n))
    return DFx
    
def kf_calcHx(t, x, u):
    """output dynamics equation h(x,u,t)"""
    n = np.size(x,0)
    u = x[0]
    v = x[1]
    w = x[2]
    ca = x[3]
    if n == 4:
        zpred = np.zeros((3))
        zpred[0] = np.arctan2(w,u)*(ca+1)
        zpred[1] = np.arctan2(v,np.sqrt(u**2+w**2))
        zpred[2] = np.sqrt(u**2 + v**2 + w**2)
    else:
        print('Alarm')
#    print('zpred=',zpred)
    return zpred
    
def kf_calcDHx(t, x, u):
    """Calculate Jacobian matrix of output dynamics"""
#    n = np.size(x, 0)
    Hx = np.zeros((3,4))
    u = x[0]
    v = x[1]
    w = x[2]
    ca = x[3]
    Hx[0,0] = -(ca + 1) * w/(u**2+w**2)
    Hx[0,1] = 0.
    Hx[0,2] = (ca + 1) * u/(u**2+w**2)
    Hx[0,3] = np.arctan2(w,u)
    Hx[1,0] = -((u*v)/(np.sqrt(u**2 + w**2) * (u**2 + v**2 + w**2)))
    Hx[1,1] = np.sqrt(u**2 + w**2)/(u**2 + v**2 + w**2)
    Hx[1,2] = -((v*w)/(np.sqrt(u**2 + w**2) * (u**2 + v**2 + w**2)))
    Hx[1,3] = 0.
    Hx[2,0] = u/np.sqrt(u**2+v**2+w**2)
    Hx[2,1] = v/np.sqrt(u**2+v**2+w**2)
    Hx[2,2] = w/np.sqrt(u**2+v**2+w**2)
    Hx[2,3] = 0.
    return Hx
    
def rk4(fn, xin, uin, t):
    a = t[0]
    b = t[1]
    w = xin
    N = 2
    h = (b-a) / N
    t = a
    for j in range(N):
        K1 = h * fn(t, w, uin)
        K2 = h * fn(t+h/2, w+K1/2, uin)
        K3 = h * fn(t+h/2, w+K2/2, uin)
        K4 = h * fn(t+h, w+K3, uin)

        w = w + (K1 + 2*K2 + 2*K3 + K4) / 6
        t = a + j*h
    return [t,w]
    
def c2d(a, b, t):
    """Custom c2d function"""
    n  = np.size(a, 1)
    nb = np.size(b, 1)
    temp1 = np.concatenate((a, b), axis=1)*t
    temp2 = np.zeros((nb,n+nb))

    temp = np.concatenate((temp1, temp2), 0)
    s = np.linalg.expm(temp)
    Phi = s[0:n,0:n]
    Gamma = s[0:n,n:n+nb]
    return Phi, Gamma
    
    
def calcObsRank(H, Fx):

    nstates = np.size(Fx,0)
    F = np.eye(np.size(Fx,0))
    Rank = np.zeros(nstates)
    for i in range(nstates):
        Rank = np.vstack((Rank, H.dot(F)))
        print(Rank)
        F = F.dot(Fx)
    Rank = np.vstack((Rank, H.dot(F)))
    print(Rank)
    r    = np.rank(Rank)
    return r

    
# simulation parameters

dt   = 0.01
N    = len(c_m)
n    = 4 # number of states
Ex_0 = [150., 0., 0., 0.] # initial estimate of optimal value of x_k_1k_1
#x_0  = np.ones((n)) # initial state
nm   = 3 # number of measurements
m    = 3 # number of inputs

B    = np.vstack((np.eye(3),np.zeros(3))) # input matrix
G    = np.eye(n) # noise input matrix

# System noise statistics:
#Ew   = [0., 0., 0., 0.] # bias
stdw = [1e-3, 1e-3, 1e-3, 0.] # noise variance
Q    = np.diag(np.power(stdw, 2))
#w_k  = np.reshape(np.tile(stdw, N),(4,N)) * np.randn(n, N) + np.reshape(np.tile(Ew, N),(4,N))

# Measurement noise statistics:
#Ev   = [0., 0., 0.] # bias
stdv = [0.01, 0.0058, 0.112] # noise variance
R    = np.diag(np.power(stdv, 2))
#v_k  = np.reshape(np.tile(stdv, N),(3,N)) * np.randn(nm, N) + np.reshape(np.tile(Ev, N),(3,N))

stdx_0  = [2.0, 2.0, 2.0, 50.0]
P_0     = np.diag(np.power(stdx_0, 2))


#==============================================================================
# # Real simulated state-variable and measurements data:
# x = x_0
# X_k = np.zeros((n, N))
# Z_k = np.zeros((nm, N))
# U_k = np.zeros((m, N))
# for i in range(N):
#     dx = kf_calcFx(0, x, U_k[:,i])
# #    print(dx)
#     x = x + (dx + w_k[:,i]) * dt
#     X_k[:,i] = x # store state
# #    print('zk = ', Z_k[:,i])
# #    print(x, v_k[:,i])
#     Z_k[:,i] = kf_calcHx(0, x, U_k[:,i]) + v_k[:,i] # calculate measurement 
#==============================================================================

XX_k1k1  = np.zeros((n, N))
PP_k1k1  = np.zeros((n, N))
STDx_cor = np.zeros((n, N))
z_pred   = np.zeros((nm, N))
IEKFitcount = np.zeros(N)
epsilon  = 1e-12
doIEKF   = 1
do_plot  = 0
maxIterations = 2
T = np.zeros((1,N))

x_k_1k_1 = Ex_0 # x(0|0)=E{x_0}
P_k_1k_1 = P_0 # P(0|0)=P(0)

ti = 0
tf = dt

U_k = u_k
Z_k = z_k

# Run the filter through all N samples
for k in range(N):
    # Prediction x(k+1|k) 
    [t, x_kk_1] = rk4(kf_calcFx, x_k_1k_1, U_k[:,k], [ti, tf])
#    print "this stuff", t, x_kk_1
    T[:,k] = t
    
    # z(k+1|k) (predicted output)
    z_kk_1 = kf_calcHx(0, x_kk_1, U_k[:,k])
    z_pred[:,k] = z_kk_1;

    # Calc Phi(k+1,k) and Gamma(k+1,k)
    Fx = kf_calcDFx(0, x_kk_1, U_k[:,k])
    
    [dummy, Psi] = c2d(Fx, B, dt)
    [Phi, Gamma] = c2d(Fx, G, dt)
    
    # P(k+1|k) (prediction covariance matrix)
    P_kk_1 = Phi*P_k_1k_1*np.transpose(Phi) + Gamma*Q*np.transpose(Gamma)
    P_pred = np.diag(P_kk_1)
    stdx_pred = np.sqrt(np.diag(P_kk_1))

    if doIEKF:

        # do the iterative part
        eta2    = x_kk_1
        err     = 2*epsilon

        itts    = 0
        while err > epsilon:# and itts < maxIterations:
            if (itts >= maxIterations):
#                print('Terminating IEKF: exceeded max iterations {}'.format(maxIterations))
                break
            
            itts    = itts + 1
            eta1    = eta2

            # Construct the Jacobian H = d/dx(h(x))) with h(x) the observation model transition matrix 
            Hx      = kf_calcDHx(0, eta1, U_k[:,k])
            
            # Check observability of state
            if (k == 0 and itts == 0):
                rankHF = calcObsRank(Hx, Fx)
                if (rankHF < n):
                    print('The current state is not observable; rank of Observability Matrix is %d, should be %d', rankHF, n);

            
            # The innovation matrix
            Ve = Hx.dot(P_kk_1).dot(np.transpose(Hx)) + R

            # calculate the Kalman gain matrix
            K = P_kk_1.dot(np.transpose(Hx)).dot(np.linalg.inv(Ve))
            # new observation state
            z_p     = kf_calcHx(0, eta1, U_k[:,k]) #fpr_calcYm(eta1, u)

            eta2    = x_kk_1 + K.dot(Z_k[:,k] - z_p - Hx.dot(x_kk_1 - eta1))
            err     = np.linalg.norm((eta2 - eta1), np.inf) / np.linalg.norm(eta1, np.inf)
        IEKFitcount[k]    = itts
        x_k_1k_1          = eta2

    else:
        # Correction
        Hx = kf_calcDHx(0, x_kk_1, U_k[:,k])
        # Pz(k+1|k) (covariance matrix of innovation)
        Ve = Hx.dot(P_kk_1).dot(np.transpose(Hx)) + R
    
        # K(k+1) (gain)
        K = P_kk_1.dot(np.transpose(Hx)).dot(np.linalg.inv(Ve))
        # Calculate optimal state x(k+1|k+1)
        x_k_1k_1 = x_kk_1 + K.dot(Z_k[:,k] - z_kk_1)

    # P(k+1|k+1) (correction) using the numerically stable form of
    # P_k_1k_1 = (eye(n) - K*Hx) * P_kk_1; 
    P_k_1k_1 = (np.eye(n) - K.dot(Hx)).dot(P_kk_1).dot(np.transpose(np.eye(n) -
                                    K.dot(Hx))) + K.dot(R).dot(np.transpose(K))
    
    P_cor = np.diag(P_k_1k_1)
    stdx_cor = np.sqrt(P_cor)

    # Next step
    ti = tf
    tf = tf + dt
    
    # store results
    XX_k1k1[:,k] = x_k_1k_1
#    PP_k1k1[:,k] = P_k_1k_1
    STDx_cor[:,k] = stdx_cor
    

# =============================================================================
# Creating moddeling data and validation data
# =============================================================================

m  = np.size(z_pred,0) # number of states
N  = np.size(z_pred,1) # number of measurements
meas_list = range(N) # list of measurement indices
ratio = 0.2
mod_list2 = [meas_list[int(i/ratio)] for i in range(int(N*ratio))]
val_list = [i for i in meas_list if i not in mod_list2]
mod_list2.append(meas_list[-1])
mod_list = mod_list2


# select modeling part of the data
mod_data = np.zeros((m,len(mod_list)))

mod_out  = np.zeros((len(mod_list)))
mod_out[:] = [c_m[i] for i in mod_list]

mod_time = np.zeros((len(mod_list)))
mod_time[:] = [T[0][i] for i in mod_list]

for j in range(m):
    mod_data[j,:] = [z_pred[j,i] for i in mod_list]
    

# select validation part of the data
val_data = np.zeros((len(val_list)))
val_time = np.zeros((len(val_list)))
val_time[:] = [T[0][i] for i in val_list]
val_data[:] = [c_m[i] for i in val_list]

# =============================================================================
# Store the results in a pickle for future use
# (or in other parts of this project)
# =============================================================================

import pickle

data = dict()
data['z_pred'] = z_pred
data['c_m']    = c_m
data['T']      = T[0]

data['mod_data'] = mod_data
data['mod_out']  = mod_out
data['mod_list'] = mod_list
data['mod_time'] = mod_time
data['val_data'] = val_data
data['val_list'] = val_list
data['val_time'] = val_time

with open('data.pickle', 'wb') as file:
    pickle.dump(data, file)


#==============================================================================
# PLOTTING
#==============================================================================
do_plot = 1
if do_plot:

    start = 0
    end   = -1
    
    plt.close("all")
    #plt.plot(XX_k1k1[0,:]-X_k[0,:])
    
    fig = plt.figure()
    
    ax1 = fig.add_subplot(231)
    ax1.title.set_text(r'$XX_{k1k1}$')
    ax1.plot(T[0,start:end], np.transpose(XX_k1k1[0,start:end]), label=r'$u$')
    ax1.plot(T[0,start:end], np.transpose(XX_k1k1[1,start:end]), label=r'$v$')
    ax1.plot(T[0,start:end], np.transpose(XX_k1k1[2,start:end]), label=r'$w$')
    ax1.plot(T[0,start:end], np.transpose(XX_k1k1[3,start:end]), label=r'$C_{\alpha_{up}}$')
    ax1.set_xlabel(r'$Time\,[s]$')
    #ax1.set_ylabel(r'$\beta\,[rad]$')
    ax1.legend()
    
    #ax2 = fig.add_subplot(232)
    #ax2.title.set_text(r'$X_k$')
    #ax2.plot(np.transpose(X_k))
    
    ax3 = fig.add_subplot(233)
    ax3.title.set_text(r'$STD_{cross-correlation}$')
    ax3.plot(T[0,0:500], np.transpose(STDx_cor[0,0:500]), label=r'$u$')
    ax3.plot(T[0,0:500], np.transpose(STDx_cor[1,0:500]), label=r'$v$')
    ax3.plot(T[0,0:500], np.transpose(STDx_cor[2,0:500]), label=r'$w$')
    ax3.plot(T[0,0:500], np.transpose(STDx_cor[3,0:500]), label=r'$C_{\alpha_{up}}$')
    ax3.set_ylim([-0.5,3.5])
    ax3.set_xlabel(r'$Time\,[s]$')
    ax3.set_ylabel(r'$\beta\,[rad]$')
    ax3.legend()
    
    ax4 = fig.add_subplot(234)
    ax4.title.set_text(r'$C_{\alpha_{up}}$')
    ax4.plot(T[0,start:end], XX_k1k1[3,start:end])
    ax4.set_xlabel(r'$Time\,[s]$')
    ax4.set_ylabel(r'$C_{\alpha_{up}}\,[-]$')
    
    #ax5 = fig.add_subplot(235, projection='3d')
    #ax5.scatter(alpha_m, beta_m, Vtot)
    #ax5.set_xlabel(r'$\alpha_m$')
    #ax5.set_ylabel(r'$\beta_m$')
    #ax5.set_zlabel(r'$V_{tot}$')
    
    ax6 = fig.add_subplot(232)
    ax6.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    ax6.plot(T[0,start:end], np.transpose(z_k[1,start:end]), 'g', label=r'$\beta_m$')
    ax6.plot(T[0,start:end], np.transpose(z_pred[1,start:end]), color='r', linewidth=2, label=r'$\beta_p$')
    ax6.set_xlabel(r'$Time\,[s]$')
    ax6.set_ylabel(r'$\beta\,[rad]$')
    ax6.legend()
    
    ax7 = fig.add_subplot(236)
    ax7.title.set_text(r'$Predicted\,vs\,Measured\,Output$')
    ax7.plot(T[0,start:end], np.transpose(z_k[2,start:end]), 'g', label=r'$V_{tot}$')
    ax7.plot(T[0,start:end], np.transpose(z_pred[2,start:end]), color='r', linewidth=2, label=r'$V_{tot}$')
    ax7.set_xlabel(r'$Time\,[s]$')
    ax7.set_ylabel(r'$V_{tot}\,[m/s]$')
    ax7.legend(shadow=True)
    
    ax8 = fig.add_subplot(235)
    ax8.title.set_text(r'Predicted vs Measured Output')
    ax8.plot(T[0,start:end], np.transpose(z_k[0,start:end]), label=r'$\alpha_m$')
    ax8.plot(T[0,start:end], np.transpose(z_pred[0,start:end]), label=r'$\alpha_p$')
    ax8.plot(T[0,start:end], np.transpose(z_k[0,start:end]-XX_k1k1[3,start:end]), label=r'$\alpha_{corrected}$')
    ax8.legend()
    
    #ax9 = fig.add_subplot(235)
    #ax9.title.set_text(r'$IEKF\,iterations\,per\,sample$')
    #ax9.plot(T[0,start:end], IEKFitcount[start:end], label=r'IEKF count')
    #ax9.set_ylim([-0.5, max(IEKFitcount)+0.5])
    #ax9.set_xlabel(r'$Time\,[s]$')
    #ax9.set_ylabel(r'$\beta\,[rad]$')
    
    
    #plt.legend()
    plt.show()
    
#    cwd = os.getcwd()
#    path_to_report = os.path.join(os.path.dirname(os.getcwd()), 'Report', 'figures')
#    plt.savefig(path_to_report, format='eps', dpi=1000)
