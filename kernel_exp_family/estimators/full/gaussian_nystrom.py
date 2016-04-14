from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.estimators.full.develop.gaussian_nystrom import log_pdf_naive,\
    nystrom_naive
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np

def nystrom(X, sigma, lmbda, inds):
    A, b = build_system(X, sigma, lmbda)
    
    inds_with_xi = np.zeros(len(inds)+1)
    inds_with_xi[1:] = inds+1
    
    A_mm = A[:, inds_with_xi][inds_with_xi]
    A_nm = A[:, inds_with_xi]
    b_m = b[inds_with_xi]
    
    return A_mm, A_nm, b_m, inds

def fit(X, sigma, lmbda, inds):
    N,D = X.shape
    A_nm, b = nystrom(X, sigma, lmbda, inds)
    
    A = np.dot(A_nm.T, A_nm)
    b = np.dot(A_nm.T, b).flatten()
    
    x = np.linalg.solve(A, b)
    alpha = x[0]
    beta = x[1:]
    return alpha, beta

class KernelExpFullNystromGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N, m):
        self.sigma = sigma
        self.lmbda = lmbda
        self.N = N
        self.D = D
        
        # initial RKHS function is flat
        self.alpha = 0
        self.beta = np.zeros(m)
        self.X = np.zeros((0, D))
        
        self.inds = np.sort(np.random.permutation(N*D)[:m])
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={0: self.N, 1: self.D})
        self.X = X
        self.alpha, self.beta = fit(self.X, self.sigma, self.lmbda, self.inds)
    
    def log_pdf(self, x):
        return log_pdf_naive(x, self.X, self.sigma, self.alpha, self.beta, self.inds)

    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        return np.zeros(self.D)

    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return 0.

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
