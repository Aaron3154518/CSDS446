# -*- coding: utf-8 -*-
"""
Script to fit stochastic block models to Facebook friendship data

@author: Kevin S. Xu
"""

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import sbm

#%% Load data and visualize adjacency matrix
adj = np.loadtxt('facebook-links-filtered-adj.txt')
net = nx.Graph(adj)
plt.figure()
plt.spy(adj)
plt.show()

#%% Estimate cluster memberships using spectral clustering
clusterId = sbm.spectralCluster(adj,directed=False)
nClusters = np.max(clusterId)+1
clusterSizes = np.histogram(clusterId, bins=nClusters)[0]
print(clusterSizes)
print(clusterId)

# Re-order nodes by class memberships and re-examine adjacency matrix
sortId = np.argsort(clusterId)
print(clusterId[sortId])
plt.figure()
plt.spy(adj[sortId[:,np.newaxis],sortId])
plt.show()

#%% Estimate edge probabilities at the block level
blockProb,logLik = sbm.estimateBlockProb(adj,clusterId,directed=False)
print(blockProb)
print(logLik)

# View estimated edge probabilities as a heat map
plt.figure()
plt.imshow(blockProb)
plt.colorbar()
plt.show()

#%% Compute transitivity of actual network using NetworkX
trans = nx.transitivity(net)
print(trans)

#%% Simulate new networks from SBM fit to check model goodness of fit
nRuns = 50
blockProbSim = np.zeros((nClusters,nClusters,nRuns))
transSim = np.zeros(nRuns)
for run in range(nRuns):
    # Simulate new adjacency matrix and create NetworkX object for it
    adjSim = sbm.generateAdj(clusterId,blockProb,directed=False)
    netSim = nx.Graph(adjSim)
    blockProbSim[:,:,run] = sbm.estimateBlockProb(adjSim,clusterId,
                                                  directed=False)[0]
    transSim[run] = nx.transitivity(netSim)
meanBlockProbSim = np.mean(blockProbSim,axis=2)
stdBlockProbSim = np.std(blockProbSim,axis=2)
print('Actual block densities:')
print(blockProb)
print('Mean simulated block densities:')
print(meanBlockProbSim)
print('95% confidence interval lower bound:')
print(meanBlockProbSim-2*stdBlockProbSim)
print('95% confidence interval upper bound:')
print(meanBlockProbSim+2*stdBlockProbSim)

# View mean simulated edge probabilities as a heat map
plt.figure()
plt.imshow(meanBlockProbSim)
plt.colorbar()
plt.show()

plt.figure()
plt.hist(transSim)
plt.title('Actual transitivity: %f' % trans)
plt.show()
