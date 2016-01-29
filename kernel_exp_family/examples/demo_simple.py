from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.estimators.lite.gaussian_low_rank import KernelExpLiteGaussianLowRank
from kernel_exp_family.examples.tools import visualise_fit
import matplotlib.pyplot as plt
import numpy as np


def get_KernelExpFiniteGaussian_instance(D):
    # arbitrary choice of parameters here
    sigma = 2
    lmbda = 1
    m = 100
    return KernelExpFiniteGaussian(sigma, lmbda, m, D)

def get_KernelExpLiteGaussian_instance(D, N):
    # arbitrary choice of parameters here
    sigma = 1.
    lmbda = 0.01
    return KernelExpLiteGaussian(sigma, lmbda, D, N)

def get_KernelExpLiteGaussianLowRank_instance(D, N):
    # arbitrary choice of parameters here
    sigma = 1.
    lmbda = 0.01
    return KernelExpLiteGaussianLowRank(sigma, lmbda, D, N, eta=.1)

class ground_truth():
    def __init__(self):
        pass
    def log_pdf(self, x):
        return -0.5 * np.dot(x, x)
    def grad(self, x):
        return -0.5 * x
    def fit(self, X):
        pass
    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    def objective(self, x):
        return 0.

if __name__ == '__main__':
    """
    This simple demo demonstrates how to use the the object-oriented API.
    We fit our model to a simple 2D Gaussian, and plot the results.
    You can play around * 5 with different estimators in the code below and see how
    they behave.
    Note that we do not cover parameter choice in this demo.
    """
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    
    # estimator API object, try different estimators here
    estimators = [
                  get_KernelExpFiniteGaussian_instance(D),
                  get_KernelExpLiteGaussian_instance(D, N),
                  get_KernelExpLiteGaussianLowRank_instance(D, N),
                  ground_truth()
                  ]
    
    for est in estimators:
        est.fit(X)
        
        # main interface for log pdf and gradient
        print est.log_pdf_multiple(np.random.randn(2, 2))
        print est.log_pdf(np.zeros(D))
        print est.grad(np.zeros(D))
        
        # score matching objective function (can be used for parameter tuning)
        print est.objective(X)
        
        visualise_fit(est, X)
        plt.suptitle("Estimated with %s" % str(est.__class__.__name__))
    
    plt.show()
