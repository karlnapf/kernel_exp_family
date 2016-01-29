from choldate._choldate import cholupdate

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.kernels.kernels import rff_feature_map, rff_feature_map_single, \
    rff_sample_basis, rff_feature_map_grad_single, theano_available
from kernel_exp_family.tools.assertions import assert_array_shape
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
        
        # zero actual data, different from n_with_fake data below
        self.n = 0
    
        self.b, self.L_C, self.n_with_fake = self._gen_initial_solution()
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
    def _gen_initial_solution(self):
        # components of linear system, stored for online updating
        b_fake = np.zeros(self.m)
        L_C_fake = np.eye(self.m) * np.sqrt(self.lmbda)
        
        # assume have observed m terms, which is needed for making the system well-posed
        # the above L_C says that the m terms had covariance self.lmbda
        # the above b says that the m terms had mean 0
        n_fake = self.m
        
        return b_fake, L_C_fake, n_fake
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        N = len(X)
        
        # initialise solution
        b_fake, L_C_fake, n_fake = self._gen_initial_solution()
        
        # "update" initial "fake" solution in the way the it is the same as repeated updating
        self.b = (b_fake * n_fake + compute_b(X, self.omega, self.u) * N) / (n_fake + N)
        C = (np.dot(L_C_fake, L_C_fake.T) * n_fake + compute_C(X, self.omega, self.u) * N) / (n_fake + N)
        self.L_C = np.linalg.cholesky(C)
        self.n_with_fake = n_fake + N
        self.n = N
        
        self.theta = fit_L_C_precomputed(self.b, self.L_C)
    
    def update_fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        for x in X:
            self.b = update_b(x, self.b, self.n_with_fake, self.omega, self.u)
            self.L_C = update_L_C(x, self.L_C, self.n_with_fake, self.omega, self.u)
            self.n_with_fake += 1
            self.n += 1
        
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
