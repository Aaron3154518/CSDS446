# -*- coding: utf-8 -*-
"""
Script to fit latent space models to Facebook friendship data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import latent_space_model as lsm

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-links-filtered-adj.txt')
net = nx.Graph(adj)
plt.figure()
plt.axis('off')
nx.draw_networkx(net,pos=nx.kamada_kawai_layout(net),node_size=100,
                 width=0.5,alpha=0.5)
plt.show()

#%% Fit latent space model using 2 latent dimensions
posEst,biasEst,logLik,optRes = lsm.estimateParams(adj,dim=2)
print(biasEst)
print(logLik)

plt.figure()
plt.axis('off')
nx.draw_networkx(net,pos=dict(enumerate(posEst)),node_size=100,
                 width=0.5,alpha=0.5)
plt.show()

#%% Compute density and transitivity of actual network using NetworkX
density = nx.density(net)
print(density)
trans = nx.transitivity(net)
print(trans)

#%% Simulate new networks from LSM fit to check model goodness of fit
nRuns = 50
densitySim = np.zeros(nRuns)
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = lsm.generateAdj(posEst,biasEst)
    netSim = nx.Graph(adjSim)
    densitySim[run] = nx.density(netSim)
    transSim[run] = nx.transitivity(netSim)
plt.figure()
plt.hist(densitySim)
plt.title('Actual density: %f' % density)
plt.show()
plt.figure()
plt.hist(transSim)
plt.title('Actual transitivity: %f' % trans)
plt.show()

