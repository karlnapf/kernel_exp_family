from abc import abstractmethod

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.estimators.full.gaussian import build_system, \
    SE_dx_dx, SE_dx, SE_dx_i_dx_i_dx_j, SE_dx_i_dx_j
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np

def get_nystrom_inds(X, basis_size):
    N, D = X.shape
    
    inds = np.sort(np.random.permutation(N * D)[:basis_size])
    return inds

def nystrom(X, sigma, lmbda, inds):
    A, b = build_system(X, sigma, lmbda)
    
    inds_with_xi = np.zeros(len(inds)+1)
    inds_with_xi[1:] = inds+1
    
    A_mm = A[:, inds_with_xi][inds_with_xi]
    A_nm = A[:, inds_with_xi]
    b_m = b[inds_with_xi]
    
    return A_mm, A_nm, b_m, inds

def fit_nystrom(X, sigma, lmbda, inds):
    A_mm, A_nm, b_m = nystrom(X, sigma, lmbda, inds)
    
    A = np.dot(A_nm.T, A_nm) + lmbda * A_mm
    b = np.dot(A_nm.T, b_m).flatten()
    
    x = np.linalg.solve(A, b)
    alpha = x[0]
    beta = x[1:]
    return alpha, beta

def ind_to_ai(ind, D):
    """
    For a given row index of the A matrix, return corresponding data and component index
    """
    return ind/D, ind%D

class KernelExpFullNystromGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, m):
        self.sigma = sigma
        self.lmbda = lmbda
        
        # initial RKHS function is flat
        self.alpha = 0
        self.beta = np.zeros(m)
        self.X = np.zeros((0, D))
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        self.fit_wrapper_()
    
    @abstractmethod
    def fit_wrapper_(self):
        self.alpha, self.beta = fit_nystrom(self.X, self.sigma, self.lmbda)
    
    def log_pdf(self, x):
        N, D = len(self.X), self.D
        all_gradients = np.zeros((N, D))
        for a in range(N):
            x_a = self.X[a, :].reshape(-1, 1)
            all_gradients[a] = np.squeeze(SE_dx_l(x.reshape(-1, 1), x_a))
        
        betasum = 0
        for beta_idx in range(len(beta_approx)):
            # this beta corresponds to a particular (a,i) pair
            a, i = ind_2_a_i(beta_idx, D)
            betasum += all_gradients[a, i] * beta_approx[beta_idx]
        
        l = np.sqrt(np.float(self.sigma) / 2)
        SE_dx_dx_l = lambda x, y : SE_dx_dx(x, y, l)
        SE_dx_l = lambda x, y: SE_dx(x, y, l)
        
        xi = 0
        betasum = 0
        for a in range(self.N):
            x_a = self.X[a, :].reshape(-1, 1)
            xi += np.sum(SE_dx_dx_l(x.reshape(-1, 1), x_a)) / self.N
            gradient_x_xa = np.squeeze(SE_dx_l(x.reshape(-1, 1), x_a))
            betasum += np.dot(gradient_x_xa, self.beta[a, :])
        
        return self.alpha * xi + betasum

    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})

        x = x.reshape(-1, 1)
        l = np.sqrt(np.float(self.sigma) / 2)

        xi_grad = 0
        betasum_grad = 0
        for a in range(self.N):
            x_a = self.X[a, :].reshape(-1, 1)

            xi_grad += np.sum(SE_dx_i_dx_i_dx_j(x, x_a, l), axis=0) / self.N
            left_arg_hessian = SE_dx_i_dx_j(x, x_a, l)
            betasum_grad += self.beta[a, :].dot(left_arg_hessian)

        return self.alpha * xi_grad + betasum_grad

    def log_pdf_multiple(self, X):
        return np.array([self.log_pdf(x) for x in X])
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        return 0.

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
