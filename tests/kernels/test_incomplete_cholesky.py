from nose.tools import assert_equal, assert_almost_equal, assert_less_equal
from numpy.testing.utils import assert_allclose

from kernel_exp_family.kernels.incomplete_cholesky import incomplete_cholesky,\
    incomplete_cholesky_new_point, incomplete_cholesky_new_points
from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np


def test_incomplete_cholesky_1():
    X = np.arange(9.0).reshape(3, 3)
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=200.)
    temp = incomplete_cholesky(X, kernel, eta=0.8, power=2)
    R, K_chol, I, W = (temp["R"], temp["K_chol"], temp["I"], temp["W"])
    K = kernel(X)
    
    
    assert_equal(len(I), 2)
    assert_equal(I[0], 0)
    assert_equal(I[1], 2)
    
    assert_equal(K_chol.shape, (len(I), len(I)))
    for i in range(len(I)):
        assert_equal(K_chol[i, i], K[I[i], I[i]])
        
    assert_equal(R.shape, (len(I), len(X)))
    assert_almost_equal(R[0, 0], 1.000000000000000)
    assert_almost_equal(R[0, 1], 0.763379494336853)
    assert_almost_equal(R[0, 2], 0.339595525644939)
    assert_almost_equal(R[1, 0], 0)
    assert_almost_equal(R[1, 1], 0.535992421608228)
    assert_almost_equal(R[1, 2], 0.940571570355992)
    
    assert_equal(W.shape, (len(I), len(X)))
    assert_almost_equal(W[0, 0], 1.000000000000000)
    assert_almost_equal(W[0, 1], 0.569858199525808)
    assert_almost_equal(W[0, 2], 0)
    assert_almost_equal(W[1, 0], 0)
    assert_almost_equal(W[1, 1], 0.569858199525808)
    assert_almost_equal(W[1, 2], 1)

def test_incomplete_cholesky_2():
    X = np.arange(9.0).reshape(3, 3)
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=8.)
    temp = incomplete_cholesky(X, kernel, eta=0.999)
    R, K_chol, I, W = (temp["R"], temp["K_chol"], temp["I"], temp["W"])
    K = kernel(X)
     
    assert_equal(len(I), 2)
    assert_equal(I[0], 0)
    assert_equal(I[1], 2)
     
    assert_equal(K_chol.shape, (len(I), len(I)))
    for i in range(len(I)):
        assert_equal(K_chol[i, i], K[I[i], I[i]])
         
    assert_equal(R.shape, (len(I), len(X)))
    assert_almost_equal(R[0, 0], 1.000000000000000)
    assert_almost_equal(R[0, 1], 0.034218118311666)
    assert_almost_equal(R[0, 2], 0.000001370959086)
    assert_almost_equal(R[1, 0], 0)
    assert_almost_equal(R[1, 1], 0.034218071400058)
    assert_almost_equal(R[1, 2], 0.999999999999060)
     
    assert_equal(W.shape, (len(I), len(X)))
    assert_almost_equal(W[0, 0], 1.000000000000000)
    assert_almost_equal(W[0, 1], 0.034218071400090)
    assert_almost_equal(W[0, 2], 0)
    assert_almost_equal(W[1, 0], 0)
    assert_almost_equal(W[1, 1], 0.034218071400090)
    assert_almost_equal(W[1, 2], 1)
     
def test_incomplete_cholesky_3():
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=200.)
    X = np.random.randn(3000, 10)
    temp = incomplete_cholesky(X, kernel, eta=0.001)
    R, K_chol, I, W = (temp["R"], temp["K_chol"], temp["I"], temp["W"])
    K = kernel(X)
     
    assert_equal(K_chol.shape, (len(I), (len(I))))
    assert_equal(R.shape, (len(I), (len(X))))
    assert_equal(W.shape, (len(I), (len(X))))
     
    assert_less_equal(np.linalg.norm(K - R.T.dot(R)), .6)
    assert_less_equal(np.linalg.norm(K - W.T.dot(K_chol.dot(W))), .6)

def test_incomplete_cholesky_check_given_rank():
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=20.)
    X = np.random.randn(300, 10)
    eta = 5
    K_chol = incomplete_cholesky(X, kernel, eta=eta)["K_chol"]
    
    assert_equal(K_chol.shape[0], eta)

def test_incomplete_cholesky_new_point():
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=200.)
    X = np.random.randn(1000, 10)
    low_rank_dim = 15
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    R, I, nu = (temp["R"], temp["I"], temp["nu"])
    
    # construct train-train kernel matrix approximation using one by one calls
    for i in range(low_rank_dim):
        r = incomplete_cholesky_new_point(X, X[i], kernel, I, R, nu)
        assert_allclose(r, R[:,i], atol=1e-1)
    

def test_incomplete_cholesky_new_points_euqals_new_point():
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=200.)
    X = np.random.randn(1000, 10)
    low_rank_dim = 15
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    R, I, nu = (temp["R"], temp["I"], temp["nu"])
    
    R_test_full = incomplete_cholesky_new_points(X, X, kernel, I, R, nu)

    # construct train-train kernel matrix approximation using one by one calls
    R_test = np.zeros(R.shape)
    for i in range(low_rank_dim):
        R_test[:, i] = incomplete_cholesky_new_point(X, X[i], kernel, I, R, nu)
        assert_allclose(R_test[:, i], R_test_full[:, i])

def test_incomplete_cholesky_asymmetric():
    kernel = lambda X, Y = None : gaussian_kernel(X, Y, sigma=1.)
    X = np.random.randn(1000, 10)
    Y = np.random.randn(100, 10)
    
    low_rank_dim = int(len(X)*0.8)
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    R, I, nu = (temp["R"], temp["I"], temp["nu"])
    
    # construct train-train kernel matrix approximation using one by one calls
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    
    assert_allclose(kernel(X, Y), R.T.dot(R_test), atol=10e-1)
