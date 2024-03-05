# -*- coding: utf-8 -*-
"""
Functions for fitting degree-corrected stochastic block models to data

@author: Kevin S. Xu
"""

import numpy as np
from sklearn.cluster import KMeans

def regularized_spectral_clustering(A, k, tau=None):
    # Compute regularized Laplacian
    deg = np.sum(A, 0)
    if tau is None:
        tau = np.mean(deg)
    deg_tau = deg + tau
    D_tau_inv_sqrt = np.diag(1./np.sqrt(deg_tau))
    L_tau = D_tau_inv_sqrt.dot(A).dot(D_tau_inv_sqrt)
    
    # Regularized spectral clustering to recover groups
    [eigenvalues,eigenvectors] = np.linalg.eigh(L_tau)
    # Sort eigenvectors in increasing order of eigenvalues
    sort_ind = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[sort_ind]
    eigenvectors = eigenvectors[:, sort_ind]
    # Keep k largest eigenvectors and perform row normalization
    top_eigenvectors = eigenvectors[:, -k:]
    for row_ind in range(top_eigenvectors.shape[0]):
        top_eigenvectors[row_ind, :] /= np.linalg.norm(
            top_eigenvectors[row_ind, :])

    # K-means step for regularized spectral clustering    
    km = KMeans(n_clusters=k)
    g = km.fit_predict(top_eigenvectors)
    
    return g

def parameter_estimation(A, g):
    # Maximum-likelihood estimates of parameters
    k = np.max(g) + 1
    deg = np.sum(A, 0)
    c = deg
    kappa = np.zeros(k)
    M = np.zeros((k, k))
    for r in range(k):
        kappa[r] = np.sum(deg[g==r])
        for s in range(k):
            M[r, s] = np.sum(A[g==r][:, g==s])
    Omega = np.sum(deg) * M / np.outer(kappa, kappa)
    
    return c, Omega
