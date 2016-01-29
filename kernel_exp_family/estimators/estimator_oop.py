from abc import abstractmethod

from kernel_exp_family.tools.assertions import assert_array_shape, \
    assert_positive_int
from kernel_exp_family.tools.xvalidation import XVal
import numpy as np


class EstimatorBase(object):
    def __init__(self, D):
        self.D = D
    
    @abstractmethod
    def fit(self, X):
        raise NotImplementedError()
    
    def log_pdf_multiple(self, X):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        raise NotImplementedError()
    
    def grad(self, x):
        raise NotImplementedError()
    
    @abstractmethod
    def objective(self, X):
        raise NotImplementedError()
    
    @abstractmethod
    def get_name(self):
        return self.__class__.__name__
    
    def xvalidate_objective(self, X, num_folds=5, num_repetitions=1):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        assert_positive_int(num_folds)
        assert_positive_int(num_repetitions)
        
        O = np.zeros((num_repetitions, num_folds))
        for i in range(num_repetitions):
            
            xval = XVal(N=len(X), num_folds=num_folds)
            for j, (train, test) in enumerate(xval):
                self.fit(X[train])
                O[i, j] = self.objective(X[test])
        
        return O
    
    def set_parameters_from_dict(self, param_dict):
        if type(param_dict) is not dict:
            raise TypeError("Given param_dict's type (%s) must be dictionary" % str(type(param_dict)))
        
        for name, value in param_dict.items():
            if name not in self.get_parameter_names():
                raise ValueError("Parameter %s not in %s" % (name, str(self.get_parameter_names())))
            
            setattr(self, name, value)
    
    @abstractmethod
    def get_parameter_names(self):
        raise NotImplementedError
    
    def get_parameters(self):
        param_dict = {}
        for name in self.get_parameter_names():
            param_dict[name] = getattr(self, name)

        return param_dict