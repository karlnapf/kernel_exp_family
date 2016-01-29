from nose import SkipTest
from nose.tools import assert_less_equal, assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.estimators.finite.develop.gaussian import compute_b_memory, \
    compute_C_memory, _objective_sym_completely_manual, \
    _objective_sym_half_manual, compute_b_weighted, compute_C_weighted
from kernel_exp_family.estimators.finite.gaussian import fit, objective, \
    compute_b, compute_C, update_b, update_C, update_L_C, \
    KernelExpFiniteGaussian
from kernel_exp_family.kernels.kernels import rff_feature_map_grad2_d, \
    rff_feature_map_grad_d, theano_available, rff_sample_basis
import numpy as np


def test_compute_b_storage_1d1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    b_manual = -rff_feature_map_grad2_d(X, omega, u, d).flatten()
    b = compute_b_memory(X, omega, u)
    assert_allclose(b_manual, b)

def test_compute_b_storage_1d2n():
    X = np.array([[1.], [2.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    b_manual = -np.mean(rff_feature_map_grad2_d(X, omega, u, d))
    b = compute_b_memory(X, omega, u)
    assert_allclose(b_manual, b)

def test_compute_C_1d1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi = rff_feature_map_grad_d(X, omega, u, d).flatten()
    C_manual = np.outer(phi, phi)
    C = compute_C_memory(X, omega, u)
    assert_allclose(C_manual, C)

def test_compute_C_1d2n():
    X = np.array([[1.], [2.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    C_manual = np.mean(rff_feature_map_grad_d(X, omega, u, d) ** 2)
    C = compute_C_memory(X, omega, u)
    assert_allclose(C_manual, C)

def test_fit():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = fit(X, omega, u)
    theta_manual = np.linalg.solve(C, b)
    assert_allclose(theta, theta_manual)

def test_objective_sym_given_b_C():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = np.random.randn(m)
    
    J = objective(X, theta, omega, u, b, C)
    J_manual = 0.5 * np.dot(theta.T, np.dot(C, theta)) - np.dot(theta, b)
    
    assert_close(J, J_manual)

def test_objective_sym_given_b_C_equals_given_nothing():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = np.random.randn(m)
    
    J = objective(X, theta, omega, u, b, C)
    J2 = objective(X, theta, omega, u)
    
    assert_close(J, J2)

def test_objective_sym_equals_completely_manual_manually():
    N = 100
    D = 3
    m = 3
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    theta = np.random.randn(m)
    
    J_manual = 0.
    for n in range(N):
        b_manual = np.zeros(m)
        C_manual = np.zeros((m, m))
        J_n_manual = 0.
        for d in range(D):
            b_term_manual = -np.sqrt(2. / m) * np.cos(np.dot(X[n], omega) + u) * (omega[d, :] ** 2)
            b_term = rff_feature_map_grad2_d(X[n], omega, u, d)
            assert_allclose(b_term_manual, b_term)
            b_manual -= b_term_manual
            J_manual += np.dot(b_term_manual, theta)
            J_n_manual += np.dot(b_term_manual, theta)
             
            c_vec_manual = -np.sqrt(2. / m) * np.sin(np.dot(X[n], omega) + u) * omega[d, :]
            c_vec = rff_feature_map_grad_d(X[n], omega, u, d)
            assert_allclose(c_vec_manual, c_vec)
            C_term = np.outer(c_vec_manual, c_vec_manual)
            C_manual += C_term
            
            # not regularised here, done afterwards
            J_manual += 0.5 * np.dot(theta, np.dot(C_term, theta))
            J_n_manual += 0.5 * np.dot(theta, np.dot(C_term, theta))
        
        b = compute_b_memory(X[n].reshape(1, m), omega, u)
        C = compute_C_memory(X[n].reshape(1, m), omega, u)
        assert_allclose(b_manual, b)
        assert_allclose(C_manual, C)
        
        # discard regularisation for these internal checks
        J_n = objective(X[n].reshape(1, m), theta, omega, u)
        J_n_2 = 0.5 * np.dot(theta, np.dot(C, theta)) - np.dot(theta, b)
        assert_allclose(J_n_2, J_n, rtol=1e-4)
        assert_allclose(J_n_manual, J_n, rtol=1e-4)
        
    J_manual /= N
    J = objective(X, theta, omega, u)

    assert_close(J, J_manual, decimal=5)

def test_objective_sym_equals_completely_manual():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    theta = np.random.randn(m)
     
    J = objective(X, theta, omega, u)
    J_manual = _objective_sym_completely_manual(X, theta, omega, u)
     
    assert_close(J_manual, J, decimal=5)

def test_objective_sym_equals_half_manual():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    theta = np.random.randn(m)
     
    J = objective(X, theta, omega, u)
    J_manual = _objective_sym_half_manual(X, theta, omega, u)
     
    assert_close(J_manual, J)

# import matplotlib.pyplot as plt
def test_fit_returns_min_1d_grid():
    N = 100
    D = 3
    m = 1
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = fit(X, omega, u)
    J = objective(X, theta, omega, u, b, C)
    
    thetas_test = np.linspace(theta - 3, theta + 3)
    Js = np.zeros(len(thetas_test))
    
    for i, theta_test in enumerate(thetas_test):
        Js[i] = objective(X, np.array([theta_test]), omega, u, b, C)
    
    
#     plt.plot(thetas_test, Js)
#     plt.plot([theta, theta], [Js.min(), Js.max()])
#     plt.title(str(theta))
#     plt.show()

    assert_almost_equal(Js.min(), J, delta=thetas_test[1] - thetas_test[0])
    assert_almost_equal(thetas_test[Js.argmin()], theta[0], delta=thetas_test[1] - thetas_test[0])

def test_fit_returns_min_random_search():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = fit(X, omega, u)
    J = objective(X, theta, omega, u, b, C)
    
    for noise in [0.0001, 0.001, 0.1, 1, 10, 100]:
        for _ in range(10):
            theta_test = np.random.randn(m) * noise + theta
            J_test = objective(X, theta_test, omega, u, b, C)
        
            assert_less_equal(J, J_test)
        
def test_compute_b_equals_compute_b_memory():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    b = compute_b(X, omega, u)
    b_storage = compute_b_memory(X, omega, u)
    assert_allclose(b, b_storage)

def test_compute_b_equals_compute_b_weighted_constant_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    log_weights = np.log(np.ones(N))
    
    b = compute_b(X, omega, u)
    b_weighted = compute_b_weighted(X, omega, u, log_weights)
    assert_allclose(b, b_weighted)

def test_compute_b_equals_compute_b_weighted_non_constant_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    weights = np.random.randn(N)
    weights = weights / np.sum(weights) * N
    log_weights = np.log(weights)
    X_weighted = np.array([X[i] * np.exp(log_weights[i]) for i in range(N)])
    
    b_manual = compute_b(X_weighted, omega, u)
    b_weighted = compute_b_weighted(X, omega, u, log_weights)
    assert_allclose(b_manual, b_weighted)

def test_compute_b_equals_compute_b_weighted_no_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    b = compute_b(X, omega, u)
    b_weighted = compute_b_weighted(X, omega, u, log_weights=None)
    assert_allclose(b, b_weighted)

def test_compute_C_equals_compute_C_weighted_constant_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    log_weights = np.log(np.ones(N))
    
    C = compute_C(X, omega, u)
    C_weighted = compute_C_weighted(X, omega, u, log_weights)
    assert_allclose(C, C_weighted)

def test_compute_C_equals_compute_C_weighted_non_constant_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    weights = np.random.randn(N)
    weights = weights / np.sum(weights) * N
    log_weights = np.log(weights)
    X_weighted = np.array([X[i] * np.exp(log_weights[i]) for i in range(N)])
    
    C_manual = compute_C(X_weighted, omega, u)
    C_weighted = compute_C_weighted(X, omega, u, log_weights)
    assert_allclose(C_manual, C_weighted)

def test_compute_C_equals_compute_C_weighted_no_weights():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C(X, omega, u)
    C_weighted = compute_C_weighted(X, omega, u, log_weights=None)
    assert_allclose(C, C_weighted)

def test_compute_C_equals_compute_C_memory():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C(X, omega, u)
    C_storage = compute_C_memory(X, omega, u)
    assert_allclose(C, C_storage, rtol=1e-4)

def test_update_b_equals_batch():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    
    b = compute_b(X, omega, u)
    b = update_b(x, b, n=N, omega=omega, u=u)
    b_batch = compute_b(np.vstack((X, x)), omega, u)
    
    assert_allclose(b, b_batch)

def test_update_C_equals_batch():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    
    C = compute_C(X, omega, u)
    C = update_C(x, C, n=N, omega=omega, u=u)
    C_batch = compute_C(np.vstack((X, x)), omega, u)
    
    assert_allclose(C, C_batch)
    
def test_update_L_C_naive_equals_batch():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    
    L_C = np.linalg.cholesky(compute_C(X, omega, u))
    L_C = update_L_C(x, L_C, n=N, omega=omega, u=u)
    L_C_batch = np.linalg.cholesky(compute_C(np.vstack((X, x)), omega, u))

    assert_allclose(L_C, L_C_batch)

def test_update_L_C_equals_batch():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    x = np.random.randn(D)
    
    L_C = np.linalg.cholesky(compute_C(X, omega, u))
    L_C = update_L_C(x, L_C, N, omega, u)
    L_C_batch = np.linalg.cholesky(compute_C(np.vstack((X, x)), omega, u))

    assert_allclose(L_C, L_C_batch)

def test_hessian_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
    sigma = 1.
    lmbda = 1.
    N = 100
    D = 2
    m = 10
    X = np.random.randn(N, D)
    
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    est.fit(X)
    est.hessian(X[0])

def test_third_order_derivative_tensor_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
    sigma = 1.
    lmbda = 1.
    N = 100
    D = 2
    m = 10
    X = np.random.randn(N, D)
    
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    est.fit(X)
    est.third_order_derivative_tensor(X[0])

def test_update_b_equals_compute_b_when_initialised_correctly():
    sigma = 1.
    N = 20
    D = 2
    m = 10
    X = np.random.randn(N, D)
    
    # basis
    omega, u = rff_sample_basis(D, m, sigma)
    
    # initial fit and update
    b_update = np.zeros(m)
    n_update = m
    for x in X:
        b_update = update_b(x, b_update, n_update, omega, u)
        n_update += 1
    
    # initial fit and batch (average of "fake" b and new observations
    b_fake = np.zeros(m)
    n_fake = m
    b_batch = (b_fake * n_fake + compute_b(X, omega, u) * N) / (n_fake + N)
    
    assert_allclose(b_update, b_batch)

def test_update_L_C_equals_compute_L_C_when_initialised_correctly():
    sigma = 1.
    lmbda = 2.
    N = 20
    D = 2
    m = 10
    X = np.random.randn(N, D)
    
    # basis
    omega, u = rff_sample_basis(D, m, sigma)
    
    # initial fit and update
    L_C_update = np.eye(m) * np.sqrt(lmbda)
    n_update = m
    for x in X:
        L_C_update = update_L_C(x, L_C_update, n_update, omega, u)
        n_update += 1
    
    # initial fit and batch (average of "fake" b and new observations
    L_C_fake = np.eye(m) * np.sqrt(lmbda)
    n_fake = m
    C_batch = (np.dot(L_C_fake, L_C_fake.T) * n_fake + compute_C(X, omega, u) * N) / (n_fake + N)
    L_C_batch = np.linalg.cholesky(C_batch)
    
    assert_allclose(L_C_update, L_C_batch)

def test_KernelExpFiniteGaussian_fit_equals_update_fit():
    sigma = 1.
    lmbda = 2.
    m = 2
    N = 1
    D = 2

    rng_state = np.random.get_state()

    np.random.seed(0)
    est_batch = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    np.random.seed(0)
    est_update = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    
    np.random.set_state(rng_state)

    assert_allclose(est_batch.b, est_update.b)
    assert_allclose(est_batch.L_C, est_update.L_C)
    assert_allclose(est_batch.n, est_update.n)
    assert_allclose(est_batch.theta, est_update.theta)
    
    X = np.random.randn(N, D)
    est_batch.fit(X)
    
    est_update.update_fit(X)
    
    assert_allclose(est_batch.b, est_update.b)
    assert_allclose(est_batch.L_C, est_update.L_C)
    assert_allclose(est_batch.n, est_update.n)
    assert_allclose(est_batch.theta, est_update.theta)

def test_KernelExpFiniteGaussian_fit_more_than_m_data_execute():
    sigma = 1.
    lmbda = 1.
    m = 2
    N = m + 1
    D = 2
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    
    X = np.random.randn(N, D)
    est.fit(X)

def test_KernelExpFiniteGaussian_fit_exactly_m_data_execute():
    sigma = 1.
    lmbda = 1.
    m = 2
    N = m
    D = 2
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    
    X = np.random.randn(N, D)
    est.fit(X)

def test_KernelExpFiniteGaussian_fit_less_than_m_data_execute():
    sigma = 1.
    lmbda = 1.
    m = 20
    N = 10
    D = 2
    est = KernelExpFiniteGaussian(sigma, lmbda, m, D)
    
    X = np.random.randn(N, D)
    est.fit(X)
