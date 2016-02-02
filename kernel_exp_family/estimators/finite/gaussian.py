from choldate._choldate import cholupdate

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.kernels.kernels import rff_feature_map, rff_feature_map_single, \
    rff_sample_basis, rff_feature_map_grad_single, theano_available
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.tools.covariance_updates import log_weights_to_lmbdas
from kernel_exp_family.tools.numerics import log_sum_exp
import numpy as np
import scipy as sp

if theano_available:
    from kernel_exp_family.kernels.kernels import rff_feature_map_comp_hessian_theano, \
    rff_feature_map_comp_third_order_tensor_theano
    
def compute_b(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections_sum += np.mean(-Phi2 * (omega[d, :] ** 2), axis=0)
        
    return -projections_sum

def update_b_L_C_weighted(X, b, L_C, log_sum_weights, log_weights, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    # transform weights into (1-\lmbda)*old_mean+ \lmbda*new_term style updates
    lmbdas = log_weights_to_lmbdas(log_sum_weights, log_weights)
    
    # first and (negative) second derivatives of rff feature map
    projection = np.dot(X, omega) + u
    Phi = np.cos(projection) * np.sqrt(2. / m)
    Phi2 = np.sin(projection) * np.sqrt(2. / m)
    
    # not needed any longer
    del projection
    
    # work on upper triangular cholesky internally
    L_R = L_C.T
    
    b_new_term = np.zeros(m)
    for i in range(N):
        # downscale L_C once for every datum
        L_R *= np.sqrt(1 - lmbdas[i])
        
        b_new_term[:] = 0
        for d in range(D):
            b_new_term += Phi[i] * (omega[d, :] ** 2)
            
            # L_C is updated D times for every datum, each with fixed lmbda
            C_new_term = Phi2[i] * omega[d, :] * np.sqrt(lmbdas[i])
            cholupdate(L_R, C_new_term)
        
        # b is updated once per datum
        b = (1 - lmbdas[i]) * b + lmbdas[i] * b_new_term
    
    # transform back to lower triangular version
    L_C = L_R.T
    
    return b, L_C

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

def fit(X, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
    
    if C is None:
        C = compute_C(X, omega, u)
    
    theta = np.linalg.solve(C, b)
    return theta

def fit_L_C_precomputed(b, L_C):
    theta = sp.linalg.cho_solve((L_C, True), b)
    return theta

def objective(X, theta, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    return 0.5 * np.dot(theta, np.dot(C, theta)) - np.dot(theta, b)

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
        
        # zero actual data, different from sum_weights data below
        # self.log_sum_weights is number of data and fake data if weights are all 1
        self.n = 0
    
        self._initialise_solution()
    
    def _initialise_solution(self):
        # components of linear system, stored for online updating
        # assume have observed fake terms, which is needed for making the system well-posed
        # the L_C says that the fake terms had covariance self.lmbda, which is a regulariser
        self.L_C = np.eye(self.m) * np.sqrt(self.lmbda)
        
        # b and the sum of weights is taken from first batch of updates to avoid scaling issues
        self.b = None
        self.log_sum_weights = None
        
        # initial solution is just a flat function
        self.theta = np.zeros(self.m)
    
    def supports_update_fit(self):
        return True
    
    def supports_weights(self):
        return True
    
    def fit(self, X, log_weights=None):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        N = len(X)
        if log_weights is None:
            log_weights = np.log(np.ones(N))
        assert_array_shape(log_weights, ndim=1, dims={0: N})
        
        # batch learning here corresponds to repeated online-learning
        self._initialise_solution()
        self.update_fit(X, log_weights)
    
    def update_fit(self, X, log_weights=None):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        N = len(X)
        
        if log_weights is None:
            log_weights = np.zeros(N)
        assert_array_shape(log_weights, ndim=1, dims={0: N})
        
        # first update: use first of X and log_weights
        if self.log_sum_weights is None:
            self.log_sum_weights = log_weights[0]
            self.b = compute_b(X[0].reshape(1, self.D), self.omega, self.u)
        
        self.b, self.L_C = update_b_L_C_weighted(X, self.b, self.L_C,
                                                 self.log_sum_weights,
                                                 log_weights,
                                                 self.omega, self.u)
        
        # update terms and weights
        self.n += len(X)
        self.log_sum_weights = log_sum_exp(list(log_weights) + [self.log_sum_weights])
        
        # finally update solution
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
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
    
    if theano_available:
        def hessian(self, x):
            """
            Computes the Hessian of the learned log-density function.
            
            WARNING: This implementation slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            H = np.zeros((self.D, self.D))
            for i, theta_i in enumerate(self.theta):
                H += theta_i * rff_feature_map_comp_hessian_theano(x, self.omega[:, i], self.u[i])
        
            # RFF is a monte carlo average, so have to normalise by np.sqrt(m) here
            return H / np.sqrt(self.m)
        
        def third_order_derivative_tensor(self, x):
            """
            Computes the third order derivative tensor of the learned log-density function.
            
            WARNING: This implementation is slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            G3 = np.zeros((self.D, self.D, self.D))
            for i, theta_i in enumerate(self.theta):
                G3 += theta_i * rff_feature_map_comp_third_order_tensor_theano(x, self.omega[:, i], self.u[i])
        
            # RFF is a monte carlo average, so have to normalise by np.sqrt(m) here
            return G3 / np.sqrt(self.m)
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        Phi = rff_feature_map(X, self.omega, self.u)
        return np.dot(Phi, self.theta)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # note we need to recompute b and C here
        return objective(X, self.theta, self.omega, self.u)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']
    
    def set_parameters_from_dict(self, param_dict):
        EstimatorBase.set_parameters_from_dict(self, param_dict)
        
        # update basis
        self.omega, self.u = rff_sample_basis(self.D, self.m, self.sigma)
