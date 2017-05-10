from abc import abstractmethod

from kernel_exp_family.estimators.estimator_oop import EstimatorBase
try:
    from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
except ImportError:
    print("Could not import BayesOptSearch.")
from kernel_exp_family.kernels.kernels import gaussian_kernel, \
    gaussian_kernel_grad, theano_available
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.tools.log import Log
import numpy as np


if theano_available:
    from kernel_exp_family.kernels.kernels import gaussian_kernel_hessian_theano, \
        gaussian_kernel_third_order_derivative_tensor_theano
                
logger = Log.get_logger()

def compute_b(X, Y, K_XY, sigma):
    assert X.shape[1] == Y.shape[1]
    assert K_XY.shape[0] == X.shape[0]
    assert K_XY.shape[1] == Y.shape[0]
    
    NX = len(X)
    D = X.shape[1]
    
    b = np.zeros(NX)
    K1 = np.sum(K_XY, 1)
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        
        s_l = x_l ** 2
        t_l = y_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_K = s_l[:, np.newaxis] * K_XY
        D_x_K = x_l[:, np.newaxis] * K_XY
        
        b += 2. / sigma * (K_XY.dot(t_l) \
                        + np.sum(D_s_K, 1) \
                        - 2 * D_x_K.dot(y_l)) - K1
    
    return b

def compute_C(X, Y, K, sigma):
    assert X.shape[1] == Y.shape[1]
    assert K.shape[0] == X.shape[0]
    assert K.shape[1] == Y.shape[0]
    
    D = X.shape[1]
    NX = X.shape[0]
    
    C = np.zeros((NX, NX))
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_KXY = x_l[:, np.newaxis] * K
        KXY_D_y = K * y_l
        KXY_T_D_x = K.T * x_l
        D_y_KXY_T = y_l[:, np.newaxis] * K.T
        
        C += (D_x_KXY - KXY_D_y).dot(KXY_T_D_x - D_y_KXY_T)
    
    return C

def fit(X, Y, sigma, lmbda, K=None, reg_f_norm=True, reg_alpha_norm=True):
        # compute kernel matrix if needed
        if K is None:
            K = gaussian_kernel(X, Y, sigma=sigma)
        
        b = compute_b(X, Y, K, sigma)
        C = compute_C(X, Y, K, sigma)

        reg_mat = np.zeros(np.shape(K))
        if reg_f_norm:
            reg_mat += K
        
        if reg_alpha_norm:
            reg_mat += np.eye(len(K))
        
        if (reg_f_norm or reg_alpha_norm) and (lmbda>0):
            C += reg_mat * lmbda
            
        # solve (potentially regularised) linear system
        a = -sigma / 2. * np.linalg.solve(C, b)
        
        return a
    
def objective(X, Y, sigma, lmbda, alpha, K=None, K_XY=None, b=None, C=None):
    if K_XY is None:
        K_XY = gaussian_kernel(X, Y, sigma=sigma)
    
    if K is None and lmbda > 0:
        if X is Y:
            K = K_XY
        else:
            K = gaussian_kernel(X, sigma=sigma)
    
    if b is None:
        b = compute_b(X, Y, K_XY, sigma)

    if C is None:
        C = compute_C(X, Y, K_XY, sigma)
    
    
    NX = len(X)
    first = 2. / (NX * sigma) * alpha.dot(b)
    if lmbda > 0:
        second = 2. / (NX * sigma ** 2) * alpha.dot(
                                                    (C + (K + np.eye(len(C))) * lmbda).dot(alpha)
                                                    )
    else:
        second = 2. / (NX * sigma ** 2) * alpha.dot((C).dot(alpha))
    J = first + second
    return J

class KernelExpLiteGaussian(EstimatorBase):
    def __init__(self, sigma, lmbda, D, N,
                 reg_f_norm=True, reg_alpha_norm=True):
        self.sigma = sigma
        self.lmbda = lmbda
        self.D = D
        self.N = N
        self.reg_f_norm = reg_f_norm
        self.reg_alpha_norm = reg_alpha_norm
        
        # initial RKHS function is flat
        self.alpha = np.zeros(0)
        self.X = np.zeros((0, D))
    
    def fit(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        # sub-sample if data is larger than previously set N
        if len(X) > self.N:
            logger.info("Sub-sampling %d/%d data." % (self.N, len(X)))
            inds = np.random.permutation(len(X))[:self.N]
            self.X = X[inds]
        else:
            self.X = np.copy(X)
            
        self.alpha = self.fit_wrapper_()
    
    @abstractmethod
    def fit_wrapper_(self):
        self.K = gaussian_kernel(self.X, sigma=self.sigma)
        return fit(X=self.X, Y=self.X, sigma=self.sigma, lmbda=self.lmbda,
                   K=self.K,
                   reg_f_norm=self.reg_f_norm, reg_alpha_norm=self.reg_alpha_norm)
    
    def log_pdf(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
        
        k = gaussian_kernel(self.X, x.reshape(1, self.D), self.sigma)[:, 0]
        return np.dot(self.alpha, k)
    
    def grad(self, x):
        assert_array_shape(x, ndim=1, dims={0: self.D})
    
        k = gaussian_kernel_grad(x, self.X, self.sigma)
        return np.dot(self.alpha, k)
    
    if theano_available:
        def hessian(self, x):
            """
            Computes the Hessian of the learned log-density function.
            
            WARNING: This implementation slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            H = np.zeros((self.D, self.D))
            for i, a in enumerate(self.alpha):
                H += a * gaussian_kernel_hessian_theano(x, self.X[i], self.sigma)
        
            return H
        
        def third_order_derivative_tensor(self, x):
            """
            Computes the third order derivative tensor of the learned log-density function.
            
            WARNING: This implementation is slow, so don't call repeatedly.
            """
            assert_array_shape(x, ndim=1, dims={0: self.D})
            
            G3 = np.zeros((self.D, self.D, self.D))
            for i, a in enumerate(self.alpha):
                G3 += a * gaussian_kernel_third_order_derivative_tensor_theano(x, self.X[i], self.sigma)
        
            return G3
    
    def log_pdf_multiple(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        k = gaussian_kernel(self.X, X, self.sigma)
        return np.dot(self.alpha, k)
    
    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
        
        return objective(self.X, X, self.sigma, self.lmbda, self.alpha, self.K)

    def get_parameter_names(self):
        return ['sigma', 'lmbda']

class KernelExpLiteGaussianAdaptive(KernelExpLiteGaussian):
    def __init__(self, sigma, lmbda, D, N,
                 num_initial_evaluations=3, num_evaluations=3, minimum_size_learning=100,
                 num_initial_evaluations_relearn=1, num_evaluations_relearn=1,
                 param_bounds={'sigma': [-3, 3]}):
        KernelExpLiteGaussian.__init__(self, sigma, lmbda, D, N)
        
        self.bo = None
        self.param_bounds = param_bounds
        self.num_initial_evaluations = num_initial_evaluations
        self.num_iter = num_evaluations
        self.minimum_size_learning = minimum_size_learning
        
        self.n_initial_relearn = num_initial_evaluations_relearn
        self.n_iter_relearn = num_evaluations_relearn
        
        self.learning_parameters = False
        
    def fit(self, X):
        # avoid infinite recursion from x-validation fit call
        if not self.learning_parameters and len(X) >= self.minimum_size_learning:
            self.learning_parameters = True
            if self.bo is None:
                logger.info("Bayesian optimisation from scratch.")
                self.bo = BayesOptSearch(self, X, self.param_bounds, num_initial_evaluations=self.num_initial_evaluations)
                best_params = self.bo.optimize(self.num_iter)
            else:
                logger.info("Bayesian optimisation using prior model.")
                self.bo.re_initialise(X, self.n_initial_relearn)
                best_params = self.bo.optimize(self.n_iter_relearn)
            
            self.set_parameters_from_dict(best_params)
            self.learning_parameters = False
            logger.info("Learnt %s" % str(self.get_parameters()))
        
        # standard fit call from superclass
        KernelExpLiteGaussian.fit(self, X)
