from choldate._choldate import cholupdate

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.kernels.kernels import rff_feature_map, rff_feature_map_single,\
    rff_sample_basis, rff_feature_map_grad_single
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np
import scipy as sp


def compute_b(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections_sum += np.mean(-Phi2 * (omega[d, :] ** 2), 0)
        
    return -projections_sum

def compute_C(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    C = np.zeros((m, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C += np.tensordot(temp, temp, [0, 0])

    return C / N

def update_b(x, b, n, omega, u):
    D = omega.shape[0]
    m = omega.shape[1]
    
    projections_sum = np.zeros(m)
    phi = rff_feature_map_single(x, omega, u)
    for d in range(D):
        # second derivative of feature map
        phi2 = -phi * (omega[d, :] ** 2)
        projections_sum -= phi2
    
    # Knuth's running average
    n += 1
    delta = projections_sum - b
    b += delta / n
    
    return b

def update_L_C(x, L_C, n, omega, u):
    D = omega.shape[0]
    assert x.ndim == 1
    assert len(x) == D
    m = 1 if np.isscalar(u) else len(u)
    N = 1
    
    # since C has a 1/n term in it
    L_C *= np.sqrt(n)
    
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    projection = np.dot(x[np.newaxis, :], omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        
        # here temp is 1xm, this costs O(m^2)
        cholupdate(L_C, temp[0])
        
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    # since the new C has a 1/(n+1) term in it
    L_C /= np.sqrt(n + 1)
    
    return L_C

def fit(X, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
    
    if C is None:
        C = compute_C(X, omega, u)
    
    theta = np.linalg.solve(C + lmbda * np.eye(len(C)), b)
    return theta

def fit_L_C_precomputed(b, L_C):
    theta = sp.linalg.cho_solve((L_C, True), b)
    return theta

def objective(X, theta, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    I = np.eye(len(theta))
    return 0.5 * np.dot(theta, np.dot(C + lmbda * I, theta)) - np.dot(theta, b)

def update_C(x, C, n, omega, u):
    D = omega.shape[0]
    assert x.ndim == 1
    assert len(x) == D
    m = 1 if np.isscalar(u) else len(u)
    N = 1
    
    C_new = np.zeros((m, m))
    projection = np.dot(x[np.newaxis, :], omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C_new += np.tensordot(temp, temp, [0, 0])
    
    # Knuth's running average
    n = n + 1
    delta = C_new - C
    C += delta / n
    
    return C

class KernelExpFiniteGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, m, D):
        self.sigma = sigma 
        self.lmbda = lmbda
        self.m = m
        self.D = D
        self.omega, self.u = rff_sample_basis(D, m, sigma)
        
        # components of linear system, stored for potential online updating
        self.b = np.zeros(m)
        self.C = np.eye(m)
        
        # number of terms
        self.n = 0
        
        # solution, initialise to flat function
        self.theta = np.zeros(m)
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        self.b = compute_b(X, self.omega, self.u)
        self.C = compute_C(X, self.omega, self.u)
        self.n = len(X)
        
        self.theta = fit(X, self.lmbda, self.omega, self.u, self.b, self.C)
    
    def update_fit(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        self.b = update_b(x, self.b, self.n, self.omega, self.u)
        self.C = update_C(x, self.C, self.n, self.omega, self.u)
        self.n += 1
        
        # TODO: this is currently O(m^3) rather than possibly O(m^2)
        # can be achieved using low-rank updates of Cholesky of C
        self.theta = fit(x, self.lmbda, self.omega, self.u, self.b, self.C)
        
    def log_pdf(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        phi = rff_feature_map_single(x, self.omega, self.u)
        return np.dot(phi, self.theta)
    
    def grad(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        
        grad = rff_feature_map_grad_single(x, self.omega, self.u)
        return np.dot(grad, self.theta)
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        Phi = rff_feature_map(X, self.omega, self.u)
        return np.dot(Phi, self.theta)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # note we need to recompute b and C here
        return objective(X, self.theta, self.lmbda, self.omega, self.u)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
    
    def set_parameters_from_dict(self, param_dict):
        EstimatorBase.set_parameters_from_dict(self, param_dict)
        
        # update basis
        self.omega, self.u = rff_sample_basis(self.D, self.m, self.sigma)