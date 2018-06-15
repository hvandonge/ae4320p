# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:07:04 2018

@author: Henk

This thing does stuff
"""
import scipy as np
import scipy.linalg as la
import matplotlib.pyplot as plt

def simnet(net, x, y, verboose=False):
    """This thing does some stuff"""

    if net.name == 'feedforward':
#        n   = np.size(x, 1)
#        V1  = [net.IW, net.b{0,0}]*[x;ones(0,n)]
#
#        #   Generating output of the hidden layer
#        Y1  = feval(net.trainFunct{0,0},V1)
#
#        #   Generating input for the output layer
#        V2  = [net.LW,net.b{1,0}]*[Y1;ones(1,n)]
#
#        #   Generating output of the hidden layer
#        Y2  = feval(net.trainFunct{1,0},V2)
        pass

    elif net.name == 'rbf':
        Nin  = net.NinB
        Nhid = net.NhidB
        Nout = net.Nout
        epoch = 0
        E_t1 = 1e8
        eta = 0.0001
        alpha = 1.0
        Etot = []
#        Wtot = np.zeros((Nhid,Nout))
        
        while epoch < 100:#net.trainParam.epochs:
            E_t0 = E_t1
#            Y2, Y1, V1 = rbf_exe(net, x)
            Y2, Y1 = rbf_exe3(net, x)
#            E_t1 = 0.5*np.sum((y - Y2)**2)
            E_t1 = np.sqrt(np.mean((y - Y2)**2))
            Etot.append(E_t1)
#            Wtot = np.append(Wtot, net.OW)
            dE_k = (y - Y2).dot(Y1)
#            if bias == True:
#                x[]
#            dE_j = np.zeros((Nin, Nhid))
#            for i in range(Nin):
#                dE_k = np.zeros((Nhid, Nout))
#                for j in range(Nhid):
#                    for k in range(Nout):
#                        temp1 = (-(y - Y2)[k])
#                        temp2 = temp1*net.OW[j,k]
##                        temp3 = temp2*-2*beta*(x[i] - net.centers[i,j])*np.exp(-beta*(net.centers[i,j] - x[i])**2)
#                        temp3 = temp2*diff_basisfunc(x[i], net.centers[i,j], net.func)
#                        dE_j[i,j] = np.sum(temp3*x[i])
#                        dE_k[j,k] = np.sum(-(y - Y2)*Y1[j])
            
#            for j in range(Nhid):
#                for k in range(Nout):
                    

            if net.trainAlg == 'trainlm':
#                net.centers = trainLM(dE_c, net.centers, E_t1)
                nIW = trainLM(dE_j, net.IW, E_t1)
                nOW = trainLM(dE_k, net.OW, E_t1)
            else:
#                dW_j = eta*dE_j
                dW_k = eta*dE_k
#                nIW = net.IW + dW_j
                nOW = net.OW + dW_k

            if E_t1 < 1.1*E_t0:
#                eta = eta * alpha
#                net.IW = nIW
                net.OW = nOW
            else:
#                eta = eta / alpha
#                net.IW = nIW
                net.OW = nOW
#            if np.absolute(np.sum(dE_j)) < 0.01 and np.absolute(np.sum(dE_k)) < 0.01:
#                break
            epoch += 1
            if verboose:
                if epoch % 10 == 0:
                    pass
        if verboose:
            plt.plot(Etot)
    ##        plt.plot(Wtot)
            plt.grid()
            plt.show()
            pass

    else:
        print('incorrect network type')
    if verboose:
        return Y2, Y1, E_t1
    else:
        return Y2, Y1, E_t1


def basisfunc(c, d, func):
    eps = np.absolute(c-d)
    if func == 'gaus':
        beta = 4.
        return np.exp(-(beta*eps)**2)
    elif func == 'iqd':
        beta = 5.
        return (1+(beta*eps)**2)**(-1)
    elif func == 'ff':
        return (1./(1.+np.exp(-c)))
    elif func == 'custom':
        return (1.-eps)

def diff_basisfunc(c, d, func):
    eps = np.absolute(c-d)
    if func == 'gaus':
        beta = 4.
        diff = (-2*beta**2 * eps * np.exp(beta**2 * (-eps**2)))
        return diff
    elif func == 'iqd':
        beta = 5.
        return  -(2*beta**2*eps)/(beta**2*eps**2 + 1)**2
    elif func == 'ff':
        return np.exp(-c)/((np.exp(-c) + 1)**2)
    elif func == 'custom':
        return -1.


def rbf_exe2(net, x):
    Nin  = net.NinB
    Nmes = np.size(x, 1)
    Nhid = net.NhidB
    Nout = net.Nout
    Y1 = np.zeros((Nhid, Nmes))
    Y2 = np.zeros((Nout, Nmes))
    for i in range(Nin):
        for j in range(Nhid):
            for k in range(Nout):
                xi = x[i]
                yi = xi
                xj = yi*net.IW[i,j]
                yj = basisfunc(xj, net.centers[i,j], func=net.func)
                Y1[j] = Y1[j] + yj
                xk = yj*net.OW[j,k]
                yk = xk
                Y2[k] = Y2[k] + yk
    return Y2, Y1

def rbf_exe3(net, x):
    A = np.zeros((np.size(x, 1), np.size(net.centers, 1)))
    #i = number of measurements
    #j = number of neurons
    #k = number of input dimensions
    
    for j in range(np.size(net.centers, 0)):
        for i in range(np.size(x, 1)):
            x_i = x[0:2,[i]]
            nu_ij = la.norm((net.IW[:,[j]])**2 * (x_i - net.centers[:,[j]])**2)
            A[i,j] = np.exp(-nu_ij)
    Y1 = A
    Y2 = (A.dot(net.OW)).T
    return Y2, Y1

def diff_rbf_exe3(net, x):
    A = np.zeros((np.size(x, 1), np.size(net.centers, 1)))
    #i = number of measurements
    #j = number of neurons
    #k = number of input dimensions
    
    for j in range(np.size(net.centers, 0)):
        for i in range(np.size(x, 1)):
            x_i = x[0:2,[i]]
            nu_ij = la.norm((net.IW[:,[j]])**2 * (x_i - net.centers[:,[j]])**2)
            A[i,j] = np.exp(-nu_ij)
    Y1 = A.T
    return Y1

def rbf_exe(net, x):
    Nin  = np.size(x, 0)
    Nmes = np.size(x, 1)
    Nhid = np.size(net.centers, 0)
    V1   = np.zeros((Nhid, Nmes))
    for i in range(Nin):
        temp1 = net.IW[:,i,None].dot(x[None,i,:])
        temp2 = np.multiply(net.IW[:,i,None], net.centers[:,i,None])
#        temp3 = np.square(temp1-(temp2)*np.ones((1,Nmes)))
        temp3 = np.square(temp1 - temp2)
        V1 = V1 + temp3

    Y1 = np.exp(-V1)
    Y2 = net.OW.dot(Y1)
    return Y2, Y1, V1

def trainLM(dW, W0, E):
    """
    Params : ``dW`` means ``dE/dW``, or ``J``\n
    ``W = Wt`` at current iteration\n
    ``E = Error`` at current iteration
    """
    J = np.reshape(dW, (1, np.size(dW)))
#    J = dW.T
    W = np.reshape(W0, (np.size(W0), 1))
#    W = W0
    
    mu = 1.0
#    dW1 = la.pinv(J.T.dot(J) + mu*np.eye(np.size(J.T.dot(J), 0)))
    dW1 = la.pinv(J.dot(J.T) + mu)
    temp1 = J.T.dot(E)
    temp2 = dW1*temp1
    temp3 = W - temp2
    W1 = np.reshape(temp3, np.shape(W0))
#    W1 = temp3
    return W1

def shape(x):
    """Prints the shape of the input"""
    print(np.shape(x))
    
if __name__ == "__main__":
    import nn    
    
#if __name__ == "__main__":
#    from rbf_data import netRBF
#    import pickle
#    with open('data.pickle', 'rb') as file:
#        state_estimation = pickle.load(file)
#    mod_out  = state_estimation['mod_out']
#    mod_data = state_estimation['mod_data']
#    x1 = mod_data[0]
#    x2 = mod_data[1]
#    bias = netRBF.biases
#    biases = np.ones((1,len(x1)))
#    Y2, Y1, E_t1 = simnet(netRBF, mod_data, mod_out, verboose=True)
    
"""
            elif net.NhidL == 2:
                dE_j = np.zeros((Nin, Nhid2))
                for i in range(Nin):
                    for j in range(Nhid2):
                        for k in range(Nhid):
                            for l in range(Nout):
                                temp1 = (-(y - Y2)[l])
                                temp2 = temp1 * net.OW[k,l]
                                temp3 = temp2 * diff_basisfunc(x[i], net.centers[j,k], net.func)
                                temp4 = temp3 * net.HW[j,k]
                                temp5 = temp4 * diff_basisfunc(x[i], net.centers[i,j], net.func)
                                dE_j[i,j] = np.sum(temp5*x[i])
                                

                dE_k = np.zeros((Nhid2, Nhid))
                for j in range(Nhid2):
                    for k in range(Nhid):
                        for l in range(Nout):
                            temp = (-(y - Y2)[l])
                            temp2 = temp*net.OW[k,l]
    #                        temp3 = temp2*-2*beta*(x[i] - net.centers[i,j])*np.exp(-beta*(net.centers[i,j] - x[i])**2)
                            temp3 = temp2*diff_basisfunc(x[j], net.centers[j,k], net.func)

                            dE_k[j,k] = np.sum(temp3*x[j])

                dE_l = np.zeros((Nhid, Nout))
                for k in range(Nhid):
                    for l in range(Nout):
                        dE_l[k,l] = np.sum(-(y - Y2)*Y1[k])


                if net.trainAlg == 'trainlm':
                    net.OW = trainLM(dE_k, net.OW, E_t1)
    #                net.IW = trainLM(dE_j, net.IW, E_t1)
                else:
                    dW_j = eta*dE_j
                    dW_k = eta*dE_k
                    dW_l = eta*dE_l
                    net.IW = net.IW + dW_j
                    net.HW = net.HW + dW_k
                    net.OW = net.OW + dW_l
"""