from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.examples.tools import visualise_fit
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    """
    This simple demo demonstrates how to select the kernel parameter for the lite
    estimator, based on a simple manual grid search, using the in-built x-validation.
    As the xvalidation is implemented in the API base class, this can be easily
    changed to other parameters or estimators.
    """
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    
    
    # create grid over sigma parameters, fixed regulariser
    log_sigmas = np.linspace(-5, 10, 20)
    lmbda = 0.001
    
    # evaluate objective function over all those parameters
    O = np.zeros(len(log_sigmas))
    O_lower = np.zeros(len(log_sigmas))
    O_upper = np.zeros(len(log_sigmas))
    
    # grid search
    for i, sigma in enumerate(log_sigmas):
        est = KernelExpLiteGaussian(np.exp(sigma), lmbda, D, N)
        
        # this is an array num_repetitions x num_folds, each containing a objective
        xval_result = est.xvalidate_objective(X, num_folds=5, num_repetitions=2)
        O[i] = np.mean(xval_result)
        O_lower[i] = np.percentile(xval_result, 10)
        O_upper[i] = np.percentile(xval_result, 90)
    
    # best parameter
    best_log_sigma = log_sigmas[np.argmin(O)]
    
    # visualisation
    plt.figure()
    plt.plot([best_log_sigma, best_log_sigma], [np.min(O), np.max(O)], 'r')
    plt.plot(log_sigmas, O, 'b-')
    plt.plot(log_sigmas, O_lower, 'b--')
    plt.plot(log_sigmas, O_upper, 'b--')
    plt.xlim([np.min(log_sigmas) - 1, np.max(log_sigmas) + 1])
    plt.xlabel("log sigma")
    plt.ylabel("objective")
    plt.title("lmbda=%.4f" % lmbda)
    plt.legend(["Best sigma", "Performance"])
    plt.legend(["Best sigma", "Performance", "80% percentile"])
    plt.tight_layout()
    
    est.sigma = np.exp(best_log_sigma)
    est.fit(X)
    visualise_fit(est, X)
    plt.show()
