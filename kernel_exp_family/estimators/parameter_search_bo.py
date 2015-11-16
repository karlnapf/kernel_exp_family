from pybo import inits, recommenders, solvers, policies
import reggie
from reggie.means._core import Mean

from kernel_exp_family.tools.log import Log
import matplotlib.pyplot as plt
import numpy as np


logger = Log.get_logger()

class GPMean(Mean):
    """
    Allows prescribing a posterior GP mean as the prior mean for
    another GP.
    
    Taken from dev-branch of reggie
    """
    def __init__(self, gp):
        super(GPMean, self).__init__()
        self._gp = gp.copy()

    def __info__(self):
        info = []
        info.append(('gp', self._gp))
        return info

    def get_mean(self, X):
        mu, _ = self._gp.predict(X)
        return mu

    def get_grad(self, X):
        return iter([])

    def get_gradx(self, X):
        _, _, dmu, _ = self._gp.predict(X, grad=True)
        return dmu


class BayesOptSearch(object):
    def __init__(self, estimator, data, param_bounds, objective_log=False,
                 objective_log_bound=1000, n_initial=6):
        self.estimator = estimator
        self.data = data
        self.param_bounds = param_bounds
        self.objective_log = objective_log
        self.log_bound = objective_log_bound
        self.n_initial = n_initial

        # parameter space dimensions correspond to sorted parameter bound keys
        self.param_names = np.sort(param_bounds.keys())
        self.bounds = np.array([param_bounds[k] for k in self.param_names])
        
        self.initialised = False

    def _init_model(self, n_initial, previous_model=None):
        logger.info("Initial fitting using %d points" % n_initial)
        
        # get initial data and some test points.
        self.X = list(inits.init_latin(self.bounds, n_initial))
        self.Y = [self._eval(x) for x in self.X]
        
        # initial values for kernel parameters, taken from pybo code
        sn2 = 1e-6
        rho = np.max(self.Y) - np.min(self.Y)
        rho = 1. if (rho < 1e-1) else rho
        ell = 0.25 * (self.bounds[:, 1] - self.bounds[:, 0])
        
        if previous_model is None:
            # use data mean as GP mean
            bias = np.mean(self.Y)
            self.model = reggie.make_gp(sn2, rho, ell, bias)
            
            # define priors if gp was created from scratch
            self.model.params['mean.bias'].set_prior('normal', bias, rho)
            self.model.params['like.sn2'].set_prior('horseshoe', 0.1)
            self.model.params['kern.rho'].set_prior('lognormal', np.log(rho), 1.)
            self.model.params['kern.ell'].set_prior('uniform', ell / 100, ell * 10)
        else:
            # if there has been a previous model, use it as mean
            like = previous_model._like
            kern = previous_model._kern
            mean = GPMean(previous_model)
            self.model = reggie.GP(like, kern, mean)
        
        # initialize the MCMC inference meta-model and add data
        self.model.add_data(self.X, self.Y)
        self.model = reggie.MCMC(self.model, n=10, burn=100)
        
        # best point so far
        self.xbest = recommenders.best_incumbent(self.model, self.bounds, self.X)
        
        self.initialised = True

    def _search_domain_to_param_dict(self, x):
        # build parameter dictionary for estimator, change from log to std space
        param_dict = {}
        for i, name in enumerate(self.param_names):
            # scalar spaces
            if type(x) is float:
                x = np.array([x])

            param_dict[name] = np.exp(x[i])
                
        return param_dict
    
    def optimize(self, num_iter=10):
        if not self.initialised:
            self._init_model(n_initial=self.n_initial)
        
        logger.info("Optimising %d iterations" % num_iter)
        for _ in range(num_iter):
            index = policies.EI(self.model, self.bounds, self.X)
            xnext, _ = solvers.solve_lbfgs(index, self.bounds)

            # observe and update model
            ynext = self._eval(xnext)
            self.model.add_data(xnext, ynext)
            self.X.append(xnext)
            self.Y.append(ynext)

            # best point so far
            self.xbest = recommenders.best_incumbent(self.model, self.bounds, self.X)
        
        return self._search_domain_to_param_dict(self.xbest)

    def re_initialise(self, new_data=None, n_initial=1):
        if not self.initialised:
            raise RuntimeError("Model needs to be optimised before re-initialisation is possible. Call optimise() method")
        
        if new_data is not None:
            self.data = new_data
        
        # sample a random posterior model
        previous_model = self.model._models[np.random.randint(self.model._n)]
        
        # use as prior mean
        self._init_model(n_initial=n_initial, previous_model=previous_model)
        
        return self._search_domain_to_param_dict(self.xbest)
            
    def _eval(self, x):
        param_dict = self._search_domain_to_param_dict(x)
        
        self.estimator.set_parameters_from_dict(param_dict)
        
        # objective wants to be minimised, but Bayesian optimisation maximises
        O = -self.estimator.xvalidate_objective(self.data)
        avg = np.mean(O)
        
        if self.objective_log:
            objective = np.log(avg + self.log_bound)
            if np.isnan(objective):
                raise RuntimeError("Objective function (%f) plus log-bound (%f) was negative. Cannot take log. Increase log_bound by at least %f to resolve."
                                   % (avg, self.log_bound, np.abs(-np.mean(O) + self.log_bound)))
        else:
            objective = avg
            
        return objective

def plot_bayesopt_model_1d(bo):
    assert len(bo.param_bounds) == 1
    
    x = np.linspace(bo.bounds[0,0], bo.bounds[0,1], 100)
    mu, s2 = bo.model.predict(x[:, None])
    
    plt.plot(x, mu, 'b-')
    lower = mu - 1.96 * np.sqrt(s2)
    upper = mu + 1.96 * np.sqrt(s2)
    plt.plot(x, lower, 'b--')
    plt.plot(x, upper, 'b--')
    plt.plot(np.ravel(bo.X), bo.Y, 'rx')
    plt.plot([bo.xbest, bo.xbest], [lower, upper], 'r-')
    plt.grid(True)
