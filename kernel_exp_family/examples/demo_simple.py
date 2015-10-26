from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.examples.tools import pdf_grid, visualise_array
import matplotlib.pyplot as plt
import numpy as np


def get_KernelExpFiniteGaussian_instance(D):
    # arbitrary choice of parameters here
    gamma = 0.5
    lmbda = 0.001
    m = 100
    return KernelExpFiniteGaussian(gamma, lmbda, m, D)

def get_KernelExpLiteGaussian_instance(D):
    # arbitrary choice of parameters here
    sigma = 1.
    lmbda = 0.01
    return KernelExpLiteGaussian(sigma, lmbda, D)

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
    D, G = pdf_grid(Xs, Ys, est)
    
    class ground_truth():
        def __init__(self):
            pass
        def log_pdf(self,x):
            return -0.5 * np.dot(x, x)
        def grad(self, x):
            return np.linalg.norm(x)
        
    D_true, G_true = pdf_grid(Xs, Ys, ground_truth())
    
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
    
    plt.tight_layout()
    plt.show()
