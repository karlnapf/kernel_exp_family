from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np


def sample_basis(D, m, gamma):
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    return omega, u

def sample_basis_rational_quadratic(D, m, alpha, beta=1., return_taus=False):
    """
    Given a Gaussian kernel of the form
    k(x,y) = \exp(-\gamma  ||x-y||^2)
           = \exp(-0.5  \tau   ||x-y||^2),
           
    where
    \gamma = 0.5  \tau,
    
    this method returns a random Fourier features basis for an infinite mixture
    of Gaussian kernels (aka rational quadratic kernel)
    k(x,y) = \int d\gamma p(\tau) k(x,y),
    where p(\tau) is a Gamma distribution
    \tau \sim \texttt{Gamma}(\tau | \alpha,\beta), parametrised with
    \alpha - shape parameter
    \beta -  mean parameter (mean=shape*scale = shape/rate)
    
    The parametrisation is such that alpha, beta correspond to the closed form RQ kernel
    k(x,y) = (1+ (||x-y||^2 \tau) / (2 \alpha)),
           = (1+ (||x-y||^2) / (2 \alpha \sigma^2)),
    where \tau = \sigma^2, which is the standard form given in textbooks.
    
    I.e. in the GPML book, Chapter 4.
    """
    
    omega = np.zeros((D, m))
    taus = np.zeros(m)
    
    # sample from mixture of Gaussians
    # where the length scales are distributed according to a Gamma
    for i in range(m):
        # each sample has a different length scale
        #     
        #     mean = shape/rate = shape * scale
        # <=> scale = mean/shape = beta/alpha
        tau = np.random.gamma(shape=alpha, scale=beta / alpha)
        taus[i] = tau
        gamma = 0.5 * tau
        omega[:, i] = gamma * np.random.randn(D)
    
    u = np.random.uniform(0, 2 * np.pi, m)
    
    if return_taus:
        return omega, u, taus
    else:
        return omega, u

def feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2. / m)

def feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2. / m)
    return projection

def feature_map_grad_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
        
    projection *= omega[d, :]
    projection *= np.sqrt(2. / m)
    return -projection

def feature_map_grad2_d(X, omega, u, d):
    Phi2 = feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return -Phi2

def feature_map_grad(X, omega, u):
    # equal to the looped version, feature_map_grad_loop
    # TODO make more effecient via vectorising
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    for d in range(D):
        projections[d, :, :] = projection
        projections[d, :, :] *= omega[d, :]
    
    projections *= -np.sqrt(2. / m)
    return projections

def feature_map_grad2(X, omega, u):
    # equal to the looped version, feature_map_grad2_loop
    # TODO make more effecient via vectorising
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    Phi2 = feature_map(X, omega, u)
    for d in range(D):
        projections[d, :, :] = -Phi2
        projections[d, :, :] *= omega[d, :] ** 2
        
    return projections

def feature_map_grad_single(x, omega, u):
    D, m = omega.shape
    grad = np.zeros((D, m))
    
    for d in range(D):
        grad[d, :] = feature_map_grad_d(x, omega, u, d)
    
    return grad

def compute_b(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = feature_map(X, omega, u)
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

def fit(X, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
    
    if C is None:
        C = compute_C(X, omega, u)
    
    theta = np.linalg.solve(C + lmbda * np.eye(len(C)), b)
    return theta
    
def objective(X, theta, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    I = np.eye(len(theta))
    return 0.5 * np.dot(theta, np.dot(C + lmbda * I, theta)) - np.dot(theta, b)

class KernelExpFiniteGaussian(EstimatorBase):
    def __init__(self, gamma, lmbda, m, D):
        self.gamma = gamma
        self.lmbda = lmbda
        self.m = m
        self.D = D
        self.omega, self.u = sample_basis(D, m, gamma)
        self.theta = None
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        self.b = compute_b(X, self.omega, self.u)
        self.C = compute_C(X, self.omega, self.u)
        
        self.theta = fit(X, self.lmbda, self.omega, self.u, self.b, self.C)
    
    def log_pdf(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        
        phi = feature_map_single(x, self.omega, self.u)
        return np.dot(phi, self.theta)
    
    def grad(self, x):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        
        grad = feature_map_grad_single(x, self.omega, self.u)
        return np.dot(grad, self.theta)
    
    def log_pdf_multiple(self, X):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        Phi = feature_map(X, self.omega, self.u)
        return np.dot(Phi, self.theta)
    
    def objective(self, X):
        if self.theta is None:
            raise RuntimeError("Model not fitted yet.")
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        return objective(X, self.theta, self.lmbda, self.omega, self.u, self.b, self.C)
