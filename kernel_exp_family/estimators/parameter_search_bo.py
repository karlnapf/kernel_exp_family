import numpy as np
from pybo import solve_bayesopt


class BayesOptSearch(object):
    def __init__(self, estimator, data, param_bounds, objective_log=False, objective_log_bound=1000):
        self.estimator = estimator
        self.data = data
        self.param_bounds = param_bounds
        self.objective_log = objective_log
        self.log_bound = log_bound
#     
#         # parameter space dimensions correspond to sorted parameter bound keys
        self.param_names = np.sort(param_bounds.keys())
        self.bounds = np.array([param_bounds[k] for k in self.param_names])
         
        self.xbest = None
        self.model = None
        self.info = None

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
        self.xbest, self.model, self.info = solve_bayesopt(self._eval, self.bounds,
                                                   niter=num_iter, model=self.model)
        
        return self._search_domain_to_param_dict(self.xbest)

    def _eval(self, x):
        param_dict = self._search_domain_to_param_dict(x)
        
        self.estimator.set_parameters_from_dict(param_dict)
        O = self.estimator.xvalidate_objective(self.data)
        
        if self.objective_log:
            objective = np.log(-np.mean(O) + self.log_bound)
            if np.isnan(objective):
                raise RuntimeError("Objective function (%f) plus log-bound (%f) was negative. Cannot take log. Increase log_bound by at least %f to resolve."
                                   % (-np.mean(O), self.log_bound, np.abs(-np.mean(O) + self.log_bound)))
        else:
            objective = -np.mean(O)
            
        return objective
