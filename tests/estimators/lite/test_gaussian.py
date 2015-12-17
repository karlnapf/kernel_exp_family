from nose import SkipTest
from nose.tools import assert_almost_equal, assert_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

import kernel_exp_family.estimators.lite.develop.gaussian as develop_gaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import kernel_exp_family.estimators.lite.gaussian as gaussian
from kernel_exp_family.kernels.kernels import gaussian_kernel, theano_available
import numpy as np


def test_compute_b_against_initial_notebook():
    D = 2
    sigma = 1.
    Z = np.random.randn(100, D)
    K = gaussian_kernel(Z, sigma=sigma)
    
    # build matrix expressions from notes
    m = Z.shape[0]
    D = Z.shape[1]
    S = Z * Z
    
    b = np.zeros(m)
    for l in np.arange(D):
        s_l = S[:, l]
        x_l = Z[:, l]
        b += 2. / sigma * (K.dot(s_l) \
                        + np.diag(s_l).dot(K).dot(np.ones(m)) \
                        - 2 * np.diag(x_l).dot(K).dot(x_l)) - K.dot(np.ones(m))
    
    b_test = develop_gaussian.compute_b_sym(Z, K, sigma)
    
    assert_allclose(b, b_test)

def test_compute_C_against_initial_notebook():
    D = 2
    sigma = 1.
    Z = np.random.randn(100, D)
    K = gaussian_kernel(Z, sigma=sigma)
    
    # build matrix expressions from notes
    m = Z.shape[0]
    D = Z.shape[1]
    
    C = np.zeros((m, m))
    for l in np.arange(D):
        x_l = Z[:, l]
        C += (np.diag(x_l).dot(K) - K.dot(np.diag(x_l))).dot(K.dot(np.diag(x_l)) - np.diag(x_l).dot(K))
    
    C_test = develop_gaussian.compute_C_sym(Z, K, sigma)
    
    assert_allclose(C, C_test)

def test_compute_b_sym_against_paper():
    sigma = 1.
    D = 1
    Z = np.random.randn(1, D)
    K = gaussian_kernel(Z, sigma=sigma)
    b = develop_gaussian.compute_b_sym(Z, K, sigma)
    
    # compute by hand, well, it's just -k since rest is zero (look at it)
    x = Z[0]
    k = K[0, 0]
    b_paper = 2. / sigma * (k * (x ** 2) + (x ** 2) * k - 2 * x * k * x) - k
    
    assert_equal(b, b_paper)

def test_compute_C_sym_against_paper():
    sigma = 1.
    D = 1
    Z = np.random.randn(1, D)
    K = gaussian_kernel(Z, sigma=sigma)
    C = develop_gaussian.compute_C_sym(Z, K, sigma)
    
    # compute by hand, well, it's just zero (look at it)
    x = Z[0]
    k = K[0, 0]
    C_paper = (x * k - k * x) * (k * x - x * k)
    
    assert_equal(C, C_paper)

def test_objective_sym_against_naive():
    sigma = 1.
    D = 2
    N = 10
    Z = np.random.randn(N, D)
    
    K = gaussian_kernel(Z, sigma=sigma)
    
    num_trials = 10
    for _ in range(num_trials):
        alpha = np.random.randn(N)
        
        J_naive_a = 0
        for d in range(D):
            for i in range(N):
                for j in range(N):
                    J_naive_a += alpha[i] * K[i, j] * \
                                (-1 + 2. / sigma * ((Z[i][d] - Z[j][d]) ** 2))
        J_naive_a *= (2. / (N * sigma))
        
        J_naive_b = 0
        for d in range(D):
            for i in range(N):
                temp = 0
                for j in range(N):
                    temp += alpha[j] * (Z[j, d] - Z[i, d]) * K[i, j]
                J_naive_b += (temp ** 2)
        J_naive_b *= (2. / (N * (sigma ** 2)))
        
        J_naive = J_naive_a + J_naive_b
        
        # compare to unregularised objective
        lmbda = 0.
        J = develop_gaussian.objective_sym(Z, sigma, lmbda, alpha, K)
        assert_close(J_naive, J)

def test_objective_against_naive():
    sigma = 1.
    D = 2
    NX = 10
    NY = 20
    X = np.random.randn(NX, D)
    Y = np.random.randn(NY, D)
    
    K_XY = gaussian_kernel(X, Y, sigma=sigma)
    
    num_trials = 10
    for _ in range(num_trials):
        alpha = np.random.randn(NX)
        
        J_naive_a = 0
        for d in range(D):
            for i in range(NX):
                for j in range(NY):
                    J_naive_a += alpha[i] * K_XY[i, j] * \
                                (-1 + 2. / sigma * ((X[i][d] - Y[j][d]) ** 2))
        J_naive_a *= (2. / (NX * sigma))
        
        J_naive_b = 0
        for d in range(D):
            for i in range(NY):
                temp = 0
                for j in range(NX):
                    temp += alpha[j] * (X[j, d] - Y[i, d]) * K_XY[j, i]
                J_naive_b += (temp ** 2)
        J_naive_b *= (2. / (NX * (sigma ** 2)))
        
        J_naive = J_naive_a + J_naive_b
        
        # compare to unregularised objective
        lmbda = 0.
        J = gaussian.objective(X, Y, sigma, lmbda, alpha, K_XY=K_XY)
        assert_close(J_naive, J)
    

def test_broadcast_diag_matrix_multiply():
    K = np.random.randn(3, 3)
    x = np.array([1, 2, 3])
    assert(np.allclose(x[:, np.newaxis] * K , np.diag(x).dot(K)))

def test_score_matching_sym_execute_only():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    # only run here
    a = develop_gaussian.fit_sym(Z, sigma, lmbda)
    assert type(a) == np.ndarray
    assert len(a.shape) == 1
    assert len(a) == len(Z)

def test_compute_b_matches_sym():
    sigma = 1.
    Z = np.random.randn(10, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    b = develop_gaussian.compute_b_sym(Z, K, sigma=sigma)
    b_sym = gaussian.compute_b(Z, Z, K, sigma=sigma)
    assert_allclose(b, b_sym)

def test_compute_C_matches_sym():
    sigma = 1.
    Z = np.random.randn(10, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    C_sym = develop_gaussian.compute_C_sym(Z, K, sigma=sigma)
    C = gaussian.compute_C(Z, Z, K, sigma=sigma)
    assert_allclose(C, C_sym)

def test_compute_C_run_asym():
    sigma = 1.
    X = np.random.randn(100, 2)
    Y = np.random.randn(100, 2)

    K_XY = gaussian_kernel(X, Y, sigma=sigma)
    _ = gaussian.compute_C(X, Y, K_XY, sigma=sigma)

def test_score_matching_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    a_sym = develop_gaussian.fit_sym(Z, sigma, lmbda)
    a = gaussian.fit(Z, Z, sigma, lmbda)
    
    assert_allclose(a, a_sym)

def test_objective_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    alpha = np.random.randn(len(Z))
    
    J_sym = develop_gaussian.objective_sym(Z, sigma, lmbda, alpha)
    J = gaussian.objective(Z, Z, sigma, lmbda, alpha)
    
    print type(J)
    print type(J_sym)
    assert_equal(J, J_sym)
    
def test_objective_matches_sym_precomputed_KbC():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    K = gaussian_kernel(Z, sigma=sigma)
    
    alpha = np.random.randn(len(Z))
    C = develop_gaussian.compute_C_sym(Z, K, sigma)
    b = develop_gaussian.compute_b_sym(Z, K, sigma)
    
    K = gaussian_kernel(Z, sigma=sigma)
    J_sym = develop_gaussian.objective_sym(Z, sigma, lmbda, alpha, K, b, C)
    J = gaussian.objective(Z, Z, sigma, lmbda, alpha, K_XY=K, b=b, C=C)
    
    assert_equal(J, J_sym)

def test_score_matching_objective_matches_sym():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    J_sym = develop_gaussian.fit_sym(Z, sigma, lmbda, K)
    J = gaussian.fit(Z, Z, sigma, lmbda, K)
    
    assert_allclose(J, J_sym)

def test_objective_sym_optimum():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    a = develop_gaussian.fit_sym(Z, sigma, lmbda, K)
    J_opt = develop_gaussian.objective_sym(Z, sigma, lmbda, a, K)
    
    for _ in range(10):
        a_random = np.random.randn(len(Z))
        J = develop_gaussian.objective_sym(Z, sigma, lmbda, a_random, K)
        assert J >= J_opt

def test_objective_sym_same_as_from_estimation():
    sigma = 1.
    lmbda = 1.
    Z = np.random.randn(100, 2)
    
    K = gaussian_kernel(Z, sigma=sigma)
    a = develop_gaussian.fit_sym(Z, sigma, lmbda, K)
    C = develop_gaussian.compute_C_sym(Z, K, sigma)
    b = develop_gaussian.compute_b_sym(Z, K, sigma)
    J = develop_gaussian.objective_sym(Z, sigma, lmbda, a, K, b, C)
    
    J2 = develop_gaussian.objective_sym(Z, sigma, lmbda, a, K)
    assert_almost_equal(J, J2)

def test_hessian_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
    sigma = 1.
    lmbda = 1.
    N = 100
    D = 2
    X = np.random.randn(N, D)
    
    est = KernelExpLiteGaussian(sigma, lmbda, D, N)
    est.fit(X)
    est.hessian(X[0])

def test_gaussian_kernel_third_order_derivative_tensor_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
    sigma = 1.
    lmbda = 1.
    N = 100
    D = 2
    X = np.random.randn(N, D)
    
    est = KernelExpLiteGaussian(sigma, lmbda, D, N)
    est.fit(X)
    est.gaussian_kernel_third_order_derivative_tensor(X[0])