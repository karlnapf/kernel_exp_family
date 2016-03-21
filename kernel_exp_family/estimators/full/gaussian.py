from abc import abstractmethod

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
from kernel_exp_family.tools.assertions import assert_array_shape
import numpy as np


def SE(x, y, l=2):
    # ASSUMES COLUMN VECTORS
    diff = x - y;
    return np.squeeze(np.exp(-np.dot(diff.T, diff) / (2 * l ** 2)))

def SE_dx(x, y, l=2):
    return SE(x, y, l) * (y - x) / l ** 2

def SE_dx_dx(x, y, l=2):
    # Doing SE(x,y,l)*((y-x)**2/l**4 - 1/l**2) does not work!
    return SE(x, y, l) * (y - x) ** 2 / l ** 4 - SE(x, y, l) / l ** 2

def SE_dx_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.eye(x.size) / l ** 2
    term2 = SE_tmp * np.dot((x - y), (x - y).T) / l ** 4
    return term1 - term2

def SE_dx_dx_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.dot((x - y) ** 2, (x - y).T) / l ** 6
    term2 = SE_tmp * 2 * np.diag((x - y)[:, 0]) / l ** 4
    term3 = SE_tmp * np.repeat((x - y), x.size, 1).T / l ** 4
    return term1 - term2 - term3

def SE_dx_dx_dy_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.dot((x - y), (x - y).T) ** 2 / l ** 8
    term2 = SE_tmp * 6 * np.diagflat((x - y) ** 2) / l ** 6  # diagonal (x-y)
    term3 = (1 - np.eye(x.size)) * SE_tmp * np.repeat((x - y), x.size, 1) ** 2 / l ** 6  # (x_i-y_i)^2 off-diagonal 
    term4 = (1 - np.eye(x.size)) * SE_tmp * np.repeat((x - y).T, x.size, 0) ** 2 / l ** 6  # (x_j-y_j)^2 off-diagonal
    term5 = SE_tmp * (1 + 2 * np.eye(x.size)) / l ** 4
    
    return term1 - term2 - term3 - term4 + term5

def compute_h(kernel_dx_dx_dy, data):
    n, d = data.shape
    h = np.zeros((n, d))
    for b in range(n):
        for a in range(n):
            h[b, :] += np.sum(kernel_dx_dx_dy(data[a, :].reshape(-1, 1), data[b, :].reshape(-1, 1)), axis=0)
            
    return h / n

def compute_G(kernel_dx_dy, data):
    n, d = data.shape
    G = np.zeros((n, n, d, d))
    for a in range(n):
        for b in range(n):
            x = data[a, :].reshape(-1, 1)
            y = data[b, :].reshape(-1, 1)
            G[a, b, :, :] = kernel_dx_dy(x, y)
            
    return G

# compute_G(SE_dx_dy, test)

def compute_xi_norm_2(kernel_dx_dx_dy_dy, data):
    n, _ = data.shape
    norm_2 = 0
    for a in range(n):
        for b in range(n):
            x = data[a, :].reshape(-1, 1)
            y = data[b, :].reshape(-1, 1)
            norm_2 += np.sum(kernel_dx_dx_dy_dy(x, y))
            
    return norm_2 / n ** 2

def fit(X, sigma, lmbda):
    # esben parametrised kernel in terms of l as exp(-||---|| / (2*l^2)
    # therefore sigma = 2*(l**2)
    l = np.sqrt(np.float(sigma) / 2)
    
    n, d = X.shape
    
    SE_dx_dx_dy_l = lambda x, y: SE_dx_dx_dy(x, y, l)
    SE_dx_dy_l = lambda x, y: SE_dx_dy(x, y, l)
    SE_dx_dx_dy_dy_l = lambda x, y: SE_dx_dx_dy_dy(x, y, l)
    
    h = compute_h(SE_dx_dx_dy_l, X)
    G = compute_G(SE_dx_dy_l, X)
    xi_norm_2 = compute_xi_norm_2(SE_dx_dx_dy_dy_l, X)
    
    A = np.zeros((n * d + 1, n * d + 1))
    
    # Top left element
    A[0, 0] = np.sum(h ** 2) / n + lmbda * xi_norm_2
    
    # First row and first column
    for b in range(n):
        for j in range(d):
            A[0, 1 + b * d + j] = np.sum(G[:, b, :, j] * h) / n + lmbda * h[b, j]
            A[1 + b * d + j, 0] = A[0, 1 + b * d + j]
    
    # All other elements - (n*d)x(n*d) lower right submatrix
    for a in range(n):
        for i in range(d):
            for b in range(n):
                for j in range(d):
                    A[1 + b * d + j, 1 + a * d + i] = np.sum(G[a, :, i, :] * G[:, b, :, j]) / n + lmbda * G[a, b, i, j]
    
    b = np.zeros((n * d + 1, 1))

    b[0] = -xi_norm_2
    for a in range(n):
        for i in range(d):
            b[1 + a * d + i] = -h[a, i]
            
    x = np.linalg.solve(A, b)
    alpha = x[0]
    beta = x[1:].reshape(n, d)
    return alpha, beta

class KernelExpFullGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.N = N
        
        # initial RKHS function is flat
        self.alpha = 0
        self.beta = np.zeros(D * N)
        self.X = np.zeros((0, D))
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # sub-sample if data is larger than previously set N
        if len(X) > self.N:
            inds = np.random.permutation(len(X))[:self.N]
            self.X = X[inds]
        else:
            self.X = np.copy(X)
            
        self.fit_wrapper_()
    
    @abstractmethod
    def fit_wrapper_(self):
        self.alpha, self.beta = fit(self.X, self.sigma, self.lmbda)
    
    def log_pdf(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        l = np.sqrt(np.float(self.sigma) / 2)
        SE_dx_dx_l = lambda x, y : SE_dx_dx(x, y, l)
        SE_dx_l = lambda x, y: SE_dx(x, y, l)
        
        xi = 0
        betasum = 0
        for a in range(self.N):
            x_a = self.X[a, :].reshape(-1, 1)
            xi += np.sum(SE_dx_dx_l(x.reshape(-1, 1), x_a)) / self.N
            betasum = np.sum(SE_dx_l(x.reshape(-1, 1), x_a) * self.beta[a, :])
        
        return self.alpha * xi + betasum
    
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
