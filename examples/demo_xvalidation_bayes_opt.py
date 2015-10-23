from examples.tools import pdf_grid, visualise_array
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    """
    This simple demo demonstrates how to select the kernel parameter for the lite
    estimator, based on a Bayesian optimisation black-box optimiser.
    """
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    data = np.random.randn(200, D)
    est = KernelExpLiteGaussian(sigma=1, lmbda=.001, D=D)
    
    # specify bounds of parameters to search for
    param_bounds = {
#               'lmbda': [-3,-1], # fixed lmbda, uncomment to include in search
              'sigma': [-3,8],
              }
    
    # oop interface for optimising and using results
    # objective is not put through log here, if it is, might want to bound away from zero
    bo = BayesOptSearch(est, data, param_bounds, objective_log=False, objective_log_bound=100)
    best_params =  bo.optimize(num_iter=10)
    est.set_parameters_from_dict(best_params)
    
    # visualise found fit
    plt.figure()
    Xs = np.linspace(-5, 5)
    Ys = np.linspace(-5, 5)
    D, G = pdf_grid(Xs, Ys, est)
    
    plt.subplot(121)
    visualise_array(Xs, Ys, D, X)
    plt.title("estimate log pdf")
    
    plt.subplot(122)
    visualise_array(Xs, Ys, G, X)
    plt.title("estimate gradient norm")
    
    plt.suptitle("Best fit %s\nOptimised over: %s" % 
                 (str(est.get_parameters()), str(param_bounds)))
    plt.tight_layout()
    
    
    # plot cost function learned through Bayesian optimisation, if 1D
    if len(param_bounds) == 1:
        bounds = param_bounds[param_bounds.keys()[0]]
        x = np.linspace(bounds[0], bounds[1], 500)
        mu, s2 = bo.model.predict(x[:, None])
        
        plt.figure()
        plt.plot(x, mu, 'b-')
        lower = mu-1.96*np.sqrt(s2)
        upper = mu+1.96*np.sqrt(s2)
        plt.plot(x, lower, 'b--')
        plt.plot(x, upper, 'b--')
        plt.plot(np.ravel(bo.info.x), bo.info.y, 'rx')
        plt.plot([bo.xbest, bo.xbest], [lower, upper], 'r-')
        plt.title("objective model")
        plt.grid(True)
    
    plt.show()
