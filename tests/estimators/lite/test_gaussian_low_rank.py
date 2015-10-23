from nose.tools import timed, assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.kernels.incomplete_cholesky import incomplete_cholesky_gaussian, \
    incomplete_cholesky_new_points, incomplete_cholesky
from kernel_exp_family.kernels.kernels import gaussian_kernel

import kernel_exp_family.estimators.lite.gaussian as gaussian
import kernel_exp_family.estimators.lite.develop.gaussian as develop_gaussian
import kernel_exp_family.estimators.lite.gaussian_low_rank as gaussian_low_rank
import kernel_exp_family.estimators.lite.develop.gaussian_low_rank as develop_gaussian_low_rank


import numpy as np

def test_compute_b_sym_matches_full():
    sigma = 1.
    Z = np.random.randn(100, 2)
    low_rank_dim = int(len(Z) * .9)
    K = gaussian_kernel(Z, sigma=sigma)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=low_rank_dim)["R"]
    
    x = develop_gaussian.compute_b_sym(Z, K, sigma)
    y = develop_gaussian_low_rank.compute_b_sym(Z, R.T, sigma)
    assert_allclose(x, y, atol=5e-1)

def test_compute_b_matches_full():
    sigma = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(50, 2)
    
    low_rank_dim = int(len(X) * 0.9)
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    K_XY = kernel(X, Y)
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    
    x = gaussian.compute_b(X, Y, K_XY, sigma)
    y = gaussian_low_rank.compute_b(X, Y, R.T, R_test.T, sigma)
    assert_allclose(x, y, atol=5e-1)

def test_compute_b_matches_sym():
    sigma = 1.
    X = np.random.randn(10, 2)
    R = incomplete_cholesky_gaussian(X, sigma, eta=0.1)["R"]
    
    x = gaussian_low_rank.compute_b(X, X, R.T, R.T, sigma)
    y = develop_gaussian_low_rank.compute_b_sym(X, R.T, sigma)
    assert_allclose(x, y)
    
def test_apply_C_left_sym_matches_full():
    sigma = 1.
    N = 10
    Z = np.random.randn(N, 2)
    K = gaussian_kernel(Z, sigma=sigma)
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    
    v = np.random.randn(Z.shape[0])
    lmbda = 1.
    
    x = (develop_gaussian.compute_C_sym(Z, K, sigma) + lmbda * (K + np.eye(len(K)))).dot(v)
    y = develop_gaussian_low_rank.apply_left_C_sym(v, Z, R.T, lmbda)
    assert_allclose(x, y, atol=1e-2, rtol=1e-2)

def test_apply_C_left_matches_full():
    sigma = 1.
    N = 100
    X = np.random.randn(N, 2)
    Y = np.random.randn(20, 2)
    low_rank_dim = int(len(X) * .9)
    kernel = lambda X, Y = None: gaussian_kernel(X, Y, sigma=sigma)
    K_XY = kernel(X, Y)
    K = kernel(X)
    
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    
    v = np.random.randn(X.shape[0])
    lmbda = 1.
    
    x = (gaussian.compute_C(X, Y, K_XY, sigma) + (K + np.eye(len(X))) * lmbda).dot(v)
    y = gaussian_low_rank.apply_left_C(v, X, Y, R.T, R_test.T, lmbda)
    assert_allclose(x, y, atol=1e-1)

def apply_C_matches_sym():
    sigma = 1.
    N_X = 100
    X = np.random.randn(N_X, 2)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    temp = incomplete_cholesky(X, kernel, eta=0.1)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    
    R_test = incomplete_cholesky_new_points(X, X, kernel, I, R, nu)
    
    v = np.random.randn(N_X.shape[0])
    lmbda = 1.
    
    x = gaussian_low_rank.apply_left_C(v, X, X, R.T, R_test.T, lmbda)
    y = develop_gaussian_low_rank.apply_left_C_sym(v, X, R.T, lmbda)
    assert_allclose(x, y)
    
def test_compute_C_matches_sym():
    sigma = 1.
    Z = np.random.randn(10, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    C_sym = develop_gaussian.compute_C_sym(Z, K, sigma=sigma)
    C = gaussian.compute_C(Z, Z, K, sigma=sigma)
    assert_allclose(C, C_sym)

def test_score_matching_sym_matches_full():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    a = develop_gaussian.fit_sym(Z, sigma, lmbda)
    
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    a_cholesky_cg = develop_gaussian_low_rank.fit_sym(Z, sigma, lmbda, L=R.T)
    assert_allclose(a, a_cholesky_cg, atol=3)

def test_score_matching_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    low_rank_dim = int(len(Z) * .9)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    
    temp = incomplete_cholesky(Z, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(Z, Z, kernel, I, R, nu)
    
    a = gaussian_low_rank.fit(Z, Z, sigma, lmbda, R.T, R_test.T)
    a_sym = develop_gaussian_low_rank.fit_sym(Z, sigma, lmbda, R.T)
    
    assert_allclose(a, a_sym)

@timed(5)
def test_score_matching_sym_time():
    sigma = 1.
    lmbda = 1.
    N = 20000
    Z = np.random.randn(N, 2)
    
    R = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"]
    develop_gaussian_low_rank.fit_sym(Z, sigma, lmbda, L=R.T, cg_tol=1e-1)

def test_objective_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    alpha = np.random.randn(len(Z))
    
    temp = incomplete_cholesky(Z, kernel, eta=0.1)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    
    R_test = incomplete_cholesky_new_points(Z, Z, kernel, I, R, nu)
    
    b = gaussian_low_rank.compute_b(Z, Z, R.T, R_test.T, sigma)
    
    J_sym = develop_gaussian_low_rank.objective_sym(Z, sigma, lmbda, alpha, R.T, b)
    J = gaussian_low_rank.objective(Z, Z, sigma, lmbda, alpha, R.T, R_test.T, b)
    
    assert_close(J, J_sym)

def test_objective_matches_full():
    sigma = 1.
    lmbda = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(10, 2)
    low_rank_dim = int(len(X) * 0.9)
    
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    alpha = np.random.randn(len(X))
    
    K_XY = kernel(X, Y)
    C = gaussian.compute_C(X, Y, K_XY, sigma)
    b = gaussian.compute_b(X, Y, K_XY, sigma)
    J_full = gaussian.objective(X, Y, sigma, lmbda, alpha, K_XY=K_XY, b=b, C=C)
    
    temp = incomplete_cholesky(X, kernel, eta=low_rank_dim)
    I, R, nu = (temp["I"], temp["R"], temp["nu"])
    R_test = incomplete_cholesky_new_points(X, Y, kernel, I, R, nu)
    b = gaussian_low_rank.compute_b(X, Y, R.T, R_test.T, sigma)
    J = gaussian_low_rank.objective(X, Y, sigma, lmbda, alpha, R.T, R_test.T, b)
    
    assert_close(J, J_full, decimal=1)

def test_objective_sym_optimum():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    L = incomplete_cholesky_gaussian(Z, sigma, eta=0.1)["R"].T
    a = develop_gaussian_low_rank.fit_sym(Z, sigma, lmbda, L)
    b = develop_gaussian_low_rank.compute_b_sym(Z, L, sigma)
    J_opt = develop_gaussian_low_rank.objective_sym(Z, sigma, lmbda, a, L, b)
    
    for _ in range(10):
        a_random = np.random.randn(len(Z))
        J = develop_gaussian_low_rank.objective_sym(Z, sigma, lmbda, a_random, L)
        assert J >= J_opt

def test_objective_sym_matches_full():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    a_opt = develop_gaussian.fit_sym(Z, sigma, lmbda, K)
    J_opt = develop_gaussian.objective_sym(Z, sigma, lmbda, a_opt, K)
    
    L = incomplete_cholesky_gaussian(Z, sigma, eta=0.01)["R"].T
    a_opt_chol = develop_gaussian_low_rank.fit_sym(Z, sigma, lmbda, L)
    J_opt_chol = develop_gaussian_low_rank.objective_sym(Z, sigma, lmbda, a_opt_chol, L)
    
    assert_almost_equal(J_opt, J_opt_chol, delta=2.)
