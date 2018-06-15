# -*- coding: utf-8 -*-
"""
Created on Wed Apr 25 16:04:32 2018

@author: Henk
"""

class structtype():
    """Wonky way of creating a Matlab structure-like object"""
    pass

netRBF = structtype()

netRBF.Nin  = 2
netRBF.Nhid = 1
netRBF.Nout = 1
netRBF.NhidL = 1
netRBF.Nhid2 = 1
netRBF.trainAlg = 'grad_desc'
#netRBF.trainAlg = 'trainlm'
netRBF.trainFunc = ['radbas', 'purelin']
netRBF.funclist  = ['gaus', 'iqd', 'ff', 'custom']
netRBF.func = netRBF.funclist[0]
netRBF.beta = 5.
netRBF.range = [[-2,1],[-2,1]]
netRBF.N_centers = [10,1]
netRBF.name = 'rbf'
#netRBF.distr = 'smart' # or None
netRBF.distr = None # or None
netRBF.biases = False
netRBF.NinB = (netRBF.Nin+1) if netRBF.biases else netRBF.Nin
netRBF.NhidB = (netRBF.Nhid+1) if netRBF.biases else netRBF.Nhid
netRBF.trainParam = structtype()
netRBF.trainParam.goal   = 0
netRBF.trainParam.epochs = 200
netRBF.trainParam.min_grad = 1e-10
netRBF.trainParam.mu     = 1e-3
netRBF.trainParam.mu_dec = 0.1
netRBF.trainParam.mu_inc = 10
netRBF.trainParam.mu_max = 1e10

#netRBF.centers = np.ones((10,2))
#netRBF.centers = np.array([[-0.900000000000000,0],
#                [-0.700000000000000,0],
#                [-0.500000000000000,0],
#                [-0.300000000000000,0],
#                [-0.100000000000000,0],
#                [0.100000000000000,0],
#                [0.300000000000000,0],
#                [0.500000000000000,0],
#                [0.700000000000000,0],
#                [0.900000000000000,0]])

#netRBF.IW = np.ones((10,2))
#netRBF.IW = np.array([[6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273],
#                      [6.24415958368273,0.624415958368273]])
#netRBF.LW = np.ones((1,10))
#netRBF.OW = np.array([[-0.165029537793434,
#                       -0.414146881712120,
#                       0.404277023582498,
#                       -0.520573644129355,
#                       0.918965241416011,
#                       -0.389075595385762,
#                       -0.690169083573831,
#                       0.111016838647954,
#                       0.581087378224464,
#                       -0.112255412824312]])