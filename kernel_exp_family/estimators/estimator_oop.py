from kernel_exp_family.tools.assertions import assert_array_shape,\
    assert_positive_int
import numpy as np
from numpy import setdiff1d


class EstimatorBase(object):
    def __init__(self, D):
        self.D = D
    
    def fit(self, X):
        raise NotImplementedError()
    
    def log_pdf_multiple(self, X):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        raise NotImplementedError()
    
    def grad(self, x):
        raise NotImplementedError()
    
    def objective(self, X):
        raise NotImplementedError()
    
    def xvalidate_objective(self, X, num_folds=5, num_repetitions=1, return_variance=False):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        assert_positive_int(num_folds)
        assert_positive_int(num_repetitions)
        
