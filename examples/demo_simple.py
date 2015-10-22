# for making examples work server
import matplotlib; matplotlib.use('Agg')

import sys
print sys.path
import os
print os.getcwd()

import kernel_exp_family
print kernel_exp_family
import kernel_exp_family.estimators
import kernel_exp_family.estimators.finite
import kernel_exp_family.estimators.finite.gaussian

from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import matplotlib.pyplot as plt
import numpy as np


def get_KernelExpFiniteGaussian_instance(D):
    # arbitrary choice of parameters here
    gamma = 0.5
    lmbda = 0.000001
    m = 100
    return KernelExpFiniteGaussian(gamma, lmbda, m, D)

def get_KernelExpLiteGaussian_instance(D):
    # arbitrary choice of parameters here
    sigma = 1.
    lmbda = 0.000001
    return KernelExpLiteGaussian(sigma, lmbda, D)

def visualise_array(Xs, Ys, A, samples=None):
    im = plt.imshow(A, origin='lower')
    im.set_extent([Xs.min(), Xs.max(), Ys.min(), Ys.max()])
    im.set_interpolation('nearest')
    im.set_cmap('gray')
    if samples is not None:
        plt.plot(samples[:, 0], samples[:, 1], 'bx')
    plt.ylim([Ys.min(), Ys.max()])
    plt.xlim([Xs.min(), Xs.max()])

if __name__ == '__main__':
    """
    This simple demo demonstrates how to use the the object-oriented API.
    We fit our model to a simple 2D Gaussian, and plot the results.
    You can play around with different estimators in the code below and see how
    they behave.
    Note that we do not cover parameter choice in this demo.
    """
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    
    # estimator API object, try different estimators here
    est = get_KernelExpFiniteGaussian_instance(D)
    est = get_KernelExpLiteGaussian_instance(D)
    est.fit(X)
    
    # main interface for log pdf and gradient
    print est.log_pdf_multiple(np.random.randn(2, 2))
    print est.log_pdf(np.zeros(D))
    print est.grad(np.zeros(D))
    
    # score matching objective function (can be used for parameter tuning)
    print est.objective(X)
    
    # compute log-pdf and gradients over a grid and visualise
    Xs = np.linspace(-5, 5)
    Ys = np.linspace(-5, 5)
    D = np.zeros((len(Xs), len(Ys)))
    G = np.zeros(D.shape)
    D_true = np.zeros(D.shape)
    G_true = np.zeros(D.shape)
    
    # this is in-efficient, log_pdf_multiple on a 2d array is faster
    for i, x in enumerate(Xs):
        for j, y in enumerate(Ys):
            point = np.array([x, y])
            D[j, i] = est.log_pdf(point)
            G[j, i] = np.linalg.norm(est.grad(point))
            
            # this is the true log-pdf of the simple Gaussian target
            D_true[j, i] = -0.5 * np.dot(point, point)
            G_true[j, i] = np.linalg.norm(point)
    
    # visualise log-pdf, gradients, and ground truth
    plt.figure(figsize=(5, 5))
    
    plt.subplot(221)
    visualise_array(Xs, Ys, D, X)
    plt.title("estimate log pdf")
    
    plt.subplot(222)
    visualise_array(Xs, Ys, G, X)
    plt.title("estimate gradient norm")
    
    plt.subplot(223)
    visualise_array(Xs, Ys, D_true, X)
    plt.title("true log pdf")
    
    plt.subplot(224)
    visualise_array(Xs, Ys, G_true, X)
    plt.title("true gradient norm")
    plt.show()
