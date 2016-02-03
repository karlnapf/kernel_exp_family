from abc import abstractmethod
from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve.iterative import bicgstab

from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.estimators.parameter_search_bo import BayesOptSearch
from kernel_exp_family.kernels.incomplete_cholesky import incomplete_cholesky_gaussian, \
    incomplete_cholesky_new_points_gaussian
from kernel_exp_family.tools.assertions import assert_array_shape
from kernel_exp_family.tools.log import Log
import numpy as np


logger = Log.get_logger()

def compute_b(X, Y, L_X, L_Y, sigma):
    assert X.shape[1] == Y.shape[1]
    assert L_X.shape[0] == X.shape[0]
    assert L_Y.shape[0] == Y.shape[0]
    
    NX = len(X)
    D = X.shape[1]
    
    b = np.zeros(NX)
    LX1 = L_X.dot(np.sum(L_Y.T, 1))
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        s_l = x_l ** 2
        t_l = y_l ** 2
        
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_LX = s_l[:, np.newaxis] * L_X
        D_x_LX = x_l[:, np.newaxis] * L_X
        
        # compute b incrementally exploiting the Cholesky factorisation of K
        b += 2. / sigma * (L_X.dot(L_Y.T.dot(t_l)) \
                        + D_s_LX.dot(np.sum(L_Y.T, 1)) \
                        - 2 * D_x_LX.dot(L_Y.T.dot(y_l))) - LX1
    
    return b

def apply_left_C(v, X, Y, L_X, L_Y, lmbda):
    assert len(v.shape) == 1
    assert len(X) == len(L_X)
    assert len(Y) == len(L_Y)
    assert L_X.shape[1] == L_Y.shape[1]
    assert X.shape[1] == Y.shape[1]
     
    N_X = X.shape[0]
    D = X.shape[1]
     
    # multiply C to v (C is a sum over d=1...D)
    result = np.zeros(N_X)
    for l in range(D):
        x_l = X[:, l]
        y_l = Y[:, l]
         
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
#         D_x_KXY = x_l[:, np.newaxis] * K
#         KXY_D_y = K * y_l
#         KXY_T_D_x = K.T * x_l
#         D_y_KXY_T =  y_l[:, np.newaxis] * K.T
#         C += (D_x_KXY - KXY_D_y).dot(KXY_T_D_x - D_y_KXY_T)
        
        D_x_LX = x_l[:, np.newaxis] * L_X
        LY_T_D_y = L_Y.T * y_l
        LX_T_D_x = L_X.T * x_l
        D_y_LY = y_l[:, np.newaxis] * L_Y
         
        # right term of product
        x = L_X.T.dot(v)
        x = D_y_LY.dot(x)
        y = LX_T_D_x.dot(v)
        y = L_Y.dot(y)
         
        # right term times v
        temp = x - y
         
        # term of product
        x = LY_T_D_y.dot(temp)
        x = L_X.dot(x)
        y = L_Y.T.dot(temp)
        y = D_x_LX.dot(y)
         
        # add both terms times v to result
        result += x - y
     
    if lmbda > 0:
        # regularise with K=L_X.dot(L_X.T)
        result += lmbda * L_X.dot(L_X.T.dot(v))
    
        # regularise with I
        result += lmbda * v
     
    return result

def fit(X, Y, sigma, lmbda, L_X, L_Y,
                                                    cg_tol=1e-3,
                                                    cg_maxiter=None,
                                                    alpha0=None):
        if cg_maxiter is None:
            # CG needs at max dimension many iterations
            cg_maxiter = L_X.shape[0]
        
        NX = X.shape[0]
        
        # set up and solve regularised linear system via bicgstab
        # this never stores an NxN matrix
        b = compute_b(X, Y, L_X, L_Y, sigma)
        matvec = lambda v:apply_left_C(v, X, Y, L_X, L_Y, lmbda)
        C_operator = LinearOperator((NX, NX), matvec=matvec, dtype=np.float64)
        
        
        # for printing number of CG iterations
        global counter
        counter = 0
        def callback(x):
            global counter
            counter += 1
        
        # start optimisation from alpha0, if present
        if alpha0 is not None:
            logger.debug("Starting bicgstab from previous alpha0")
        solution, info = bicgstab(C_operator, b, tol=cg_tol, maxiter=cg_maxiter, callback=callback, x0=alpha0)
        logger.debug("Ran bicgstab for %d iterations." % counter)
        if info > 0:
            logger.warning("Warning: CG not convergence in %.3f tolerance within %d iterations" % \
                           (cg_tol, cg_maxiter))
        a = -sigma / 2. * solution
        return a

def objective(X, Y, sigma, lmbda, alpha, L_X, L_Y, b=None):
    if b is None:
        b = compute_b(X, Y, L_X, L_Y, sigma)

    N_X = len(X)
    first = 2. / (N_X * sigma) * alpha.dot(b)
    second = 2. / (N_X * sigma ** 2) * alpha.dot(apply_left_C(alpha, X, Y, L_X, L_Y, lmbda))
    J = first + second
    return J


class KernelExpLiteGaussianLowRank(KernelExpLiteGaussian):
    def __init__(self, sigma, lmbda, D, N, eta=0.1, cg_tol=1e-3, cg_maxiter=None):
        KernelExpLiteGaussian.__init__(self, sigma, lmbda, D, N)

        self.eta = eta
        self.cg_tol = cg_tol
        self.cg_maxiter = cg_maxiter
    
    @abstractmethod
    def fit_wrapper_(self):
        self.inc_cholesky = incomplete_cholesky_gaussian(self.X, self.sigma, eta=self.eta)
        L_X = self.inc_cholesky["R"].T
        
        logger.debug("Incomplete Cholesky using rank %d/%d capturing %.3f/1.0 of the variance " % \
                     (len(self.inc_cholesky['I']), len(self.X), self.eta))
        
        # start optimisation from previous alpha
        alpha0 = self.alpha if len(self.alpha) == len(self.X) and len(self.alpha) > 0 else np.zeros(len(self.X))
        
        return fit(self.X, self.X, self.sigma, self.lmbda, L_X, L_X, self.cg_tol, self.cg_maxiter, alpha0)

    def objective(self, X):
        assert_array_shape(X, ndim=2, dims={1: self.D})
           
        L_X = self.inc_cholesky["R"].T
        L_Y = incomplete_cholesky_new_points_gaussian(self.X, X, self.sigma, self.inc_cholesky['I'], self.inc_cholesky['R'], self.inc_cholesky['nu']).T
        b = compute_b(self.X, X, L_X, L_Y, self.sigma)
        return objective(self.X, X, self.sigma, self.lmbda, self.alpha, L_X, L_Y, b)

class KernelExpLiteGaussianLowRankAdaptive(KernelExpLiteGaussianLowRank):
    def __init__(self, sigma, lmbda, D, N, eta=0.1, cg_tol=1e-3, cg_maxiter=None,
                 num_initial_evaluations=3, n_iter=3, minimum_size_learning=100,
                 n_initial_relearn=1, n_iter_relearn=1,
                 param_bounds={'sigma': [-3,3]}):
        KernelExpLiteGaussianLowRank.__init__(self, sigma, lmbda, D, N, eta, cg_tol, cg_maxiter)
        
        self.bo = None
        self.param_bounds = param_bounds
        self.n_initial = num_initial_evaluations
        self.num_iter = n_iter
        self.minimum_size_learning = minimum_size_learning
        
        self.n_initial_relearn = n_initial_relearn
        self.n_iter_relearn = n_iter_relearn
        
        self.learning_parameters = False
        
    def fit(self, X):
        # avoid infinite recursion from x-validation fit call
        if not self.learning_parameters and len(X)>=self.minimum_size_learning:
            self.learning_parameters = True
            if self.bo is None:
                logger.info("Bayesian optimisation from scratch.")
                self.bo = BayesOptSearch(self, X, self.param_bounds, n_initial=self.n_initial)
                best_params = self.bo.optimize(self.num_iter)
            else:
                logger.info("Bayesian optimisation using prior model.")
                self.bo.re_initialise(X, self.n_initial_relearn)
                best_params = self.bo.optimize(self.n_iter_relearn)
            
            self.set_parameters_from_dict(best_params)
            self.learning_parameters = False
            logger.info("Learnt %s" % str(self.get_parameters()))
        
        # standard fit call from superclass
        KernelExpLiteGaussianLowRank.fit(self, X)
