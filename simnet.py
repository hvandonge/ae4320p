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
    if np.size(x, 0) != np.size(net.range, 0):
#        print("Not correct shape")
        pass

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
        eta = -0.0001
        alpha = 1.0
        Etot = []
        nIW = net.IW
        nOW = net.OW
        
#        Wtot = np.zeros((Nhid,Nout))
        while epoch < 10:#net.trainParam.epochs:
#            print np.shape(net.centers)
            E_t0 = E_t1
#            Y2, Y1, V1 = rbf_exe(net, x)
            Y2, Y1 = rbf_exe2(net, x)
            E_t1 = 0.5*np.sum((y - Y2)**2)
            E_t1i = 0.5*(y-Y2)**2
            Etot.append(E_t1)
#            Wtot = np.append(Wtot, net.OW)

#            if bias == True:
#                x[]
#            dE_j = np.zeros((Nin, Nhid))
            dE_j = np.zeros((Nhid, 2001))
            for i in range(Nin):
                dE_k = np.zeros((Nout, 2001))
                for j in range(Nhid):
                    for k in range(Nout):
                        temp1 = (-(y - Y2)[k])
                        temp2 = temp1*net.OW[j,k]
#                        temp3 = temp2*-2*beta*(x[i] - net.centers[i,j])*np.exp(-beta*(net.centers[i,j] - x[i])**2)
                        temp3 = temp2*diff_basisfunc(x[i], net.centers[i,j], net.func)
#                        print(temp3)
#                        shape(temp3)
#                        dE_j[i,j] =s np.sum(temp3*x[i])
                        dE_j[j,:] = temp3*x[i]
                        dE_k[k,:] = -(y - Y2)*Y1[j]
            
#            for j in range(Nhid):
#                for k in range(Nout):
                    


            if net.trainAlg == 'trainlm':
#                net.centers = trainLM(dE_c, net.centers, E_t1)
#                nIW[None,0,:] = trainLM(dE_j, net.IW[None,0,:], E_t1i).T
                
#                nIW[None,1,:] = trainLM(dE_j, net.IW[None,1,:], E_t1i).T
                nIW = trainLM(dE_j, net.IW, E_t1i).T
                nOW = trainLM(dE_k, net.OW, E_t1i).T
            else:
                dW_j = eta*dE_j
                dW_k = eta*dE_k
                nIW = net.IW + dW_j
                nOW = net.OW + dW_k

            if E_t1 < 1.1*E_t0:
#                eta = eta * alpha
                net.IW = nIW
                net.OW = nOW
            else:
#                eta = eta / alpha
                net.IW = nIW
                net.OW = nOW
            if np.absolute(np.sum(dE_j)) < 0.01 and np.absolute(np.sum(dE_k)) < 0.01:
                print(np.sum(dE_j))
                print(E_t1)
                break
            epoch += 1
            if verboose:
                if epoch % 10 == 0:
#                    print(E_t1)
                    pass
        if verboose:
            plt.plot(Etot)
    ##        plt.plot(Wtot)
            plt.grid()
            plt.show()
#        print(epoch)
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
    Nin  = net.Nin
    Nmes = np.size(x, 1)
    Nhid = net.Nhid
    Nhid2 = net.Nhid2
    Nout = net.Nout
    Y1 = np.zeros((Nhid, Nmes))
    Y2 = np.zeros((Nout, Nmes))
    for i in range(Nin):
        for j in range(Nhid2):
            for k in range(Nhid):
                for l in range(Nout):
                    xi = x[i]
                    yi = xi
                    xj = yi*net.IW[i,j]
                    yj = basisfunc(xj, net.centers[i,j], func=net.func)
                    Y1[l] = Y1[l] + yj
                    xk = yj*net.HW[j,k]
                    yk = basisfunc(xk, net.centers[j,k], func=net.func)
                    xl = yk*net.OW[k,l]
                    yl = xl
                    Y2[l] = Y2[l] + yl
    return Y2, Y1

def rbf_exe(net, x):
    Nin  = np.size(x, 0)
    Nmes = np.size(x, 1)
    Nhid = np.size(net.centers, 0)
    V1   = np.zeros((Nhid, Nmes))
#    print(net.centers)
    for i in range(Nin):
        temp1 = net.IW[:,i,None].dot(x[None,i,:])
        temp2 = np.multiply(net.IW[:,i,None], net.centers[:,i,None])
#        temp3 = np.square(temp1-(temp2)*np.ones((1,Nmes)))
        temp3 = np.square(temp1 - temp2)
        V1 = V1 + temp3

    Y1 = np.exp(-V1)
    Y2 = net.OW.dot(Y1)
    return Y2, Y1, V1

def trainLM(J, W0, E):
    """
    Params : ``dW`` means ``dE/dW``, or ``J``\n
    ``W = Wt`` at current iteration\n
    ``E = Error`` at current iteration
    """
#    J = np.reshape(dW, (1, np.size(dW)))
#    J = dW
    print('+++++')
    J = J.T
    shape(J) # 1,16
    W0 = W0.T
#    shape(W0) # 1,16
    E = E.T
#    shape(E) # 1,2001
#    W = np.reshape(W0, (np.size(W0), 1))
    mu = 1.0
    dW1 = la.pinv(J.T.dot(J) + mu*np.eye(np.size(J.T.dot(J), 0))) # 16,16
    shape(dW1)
#    dW1 = la.pinv(J.dot(J.T) + mu)
    print(dW1)
    temp1 = dW1.dot(J.T) # 16, 2001
#    shape(temp1)
    temp2 = temp1.dot(E)
#    shape(temp2)
    temp3 = W0 - temp2
#    shape(temp3)
    W1 = np.reshape(temp3, np.shape(W0))
#    W1 = temp3
#    shape(W1)
    return W1

def shape(x):
    """Prints the shape of the input"""
    print(np.shape(x))
    
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
                                
                print(dE_j)
                dE_k = np.zeros((Nhid2, Nhid))
                for j in range(Nhid2):
                    for k in range(Nhid):
                        for l in range(Nout):
                            temp = (-(y - Y2)[l])
                            temp2 = temp*net.OW[k,l]
    #                        temp3 = temp2*-2*beta*(x[i] - net.centers[i,j])*np.exp(-beta*(net.centers[i,j] - x[i])**2)
                            temp3 = temp2*diff_basisfunc(x[j], net.centers[j,k], net.func)
    #                        print(temp3)
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