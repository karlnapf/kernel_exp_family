from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch,\
    plot_bayesopt_model_1d
from kernel_exp_family.examples.tools import visualise_fit_2d
import matplotlib.pyplot as plt
import numpy as np


if __name__ == '__main__':
    """
    This simple demo demonstrates how to select the kernel parameter for the lite
    estimator, based on a Bayesian optimisation black-box optimiser.
    Note that this optimiser can be "hot-started", i.e. it can be reset, but using
    the previous model as initialiser for the new optimisation, which is useful
    when the objective function changes slightly, e.g. when a new data was added
    to the kernel exponential family model.
    """
    N = 200
    D = 2
    
    # fit model to samples from a standard Gaussian
    X = np.random.randn(N, D)
    
    # use any of the below models, might have to change parameter bounds
    estimators = [
                  KernelExpFiniteGaussian(sigma=1., lmbda=1., m=N, D=D),
                  KernelExpLiteGaussian(sigma=1., lmbda=.001, D=D, N=N),
                  ]
    
    for est in estimators:
        print(est.__class__.__name__)
        
        est.fit(X)
        
        # specify bounds of parameters to search for
        param_bounds = {
    #             'lmbda': [-5,0], # fixed lmbda, uncomment to include in search
                'sigma': [-2,3],
                  }
        
        # oop interface for optimising and using results
        # objective is not put through log here, if it is, might want to bound away from zero
        bo = BayesOptSearch(est, X, param_bounds, objective_log=False, objective_log_bound=100,
                            n_initial=5)
        
        # optimisation starts here, use results and apply to model
        best_params = bo.optimize(num_iter=5)
        est.set_parameters_from_dict(best_params)
        est.fit(X)
        
        visualise_fit_2d(est, X)
        plt.suptitle("Original fit %s\nOptimised over: %s" % 
                 (str(est.get_parameters()), str(param_bounds)))
        if len(param_bounds) == 1:
            plt.figure()
            plot_bayesopt_model_1d(bo)
            plt.title("Objective")
        
        # now change data, with different length scale
        X = np.random.randn(200, D) * .1
        
        # reset optimiser, which but initialise from old model, sample 3 random point to update
        best_params = bo.re_initialise(new_data=X, n_initial=3)
        
        # this optimisation now runs on the "new" objective
        best_params = bo.optimize(num_iter=3)
        est.set_parameters_from_dict(best_params)
        est.fit(X)
        
        visualise_fit_2d(est, X)
        plt.suptitle("New fit %s\nOptimised over: %s" % 
                 (str(est.get_parameters()), str(param_bounds)))
        
        if len(param_bounds) == 1:
            plt.figure()
            plot_bayesopt_model_1d(bo)
            plt.title("New objective")
        
        plt.show()
