from nose.tools import assert_less_equal, assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.estimators.finite.gaussian import feature_map, \
    feature_map_single, feature_map_derivative_d, feature_map_derivative2_d, \
    feature_map_derivatives_loop, feature_map_derivatives, \
    feature_map_derivatives2_loop, feature_map_derivatives2, \
    feature_map_grad_single, compute_b_memory, compute_C_memory, \
    score_matching_sym, objective, _objective_sym_completely_manual, \
    _objective_sym_half_manual, compute_b, compute_C
import numpy as np


def test_feature_map():
    x = 3.
    u = 2.
    omega = 2.
    phi = feature_map_single(x, omega, u)
    phi_manual = np.cos(omega * x + u) * np.sqrt(2.)
    assert_close(phi, phi_manual)

def test_feature_map_single_equals_feature_map():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    phis = feature_map(X, omega, u)
    
    for i, x in enumerate(X):
        phi = feature_map_single(x, omega, u)
        assert_allclose(phis[i], phi)

def test_feature_map_derivative_d_1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = feature_map_derivative_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_feature_map_derivative_d_2n():
    X = np.array([[1.], [3.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = feature_map_derivative_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_feature_map_derivative2_d():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative2 = feature_map_derivative2_d(X, omega, u, d)
    phi_derivative2_manual = -feature_map(X, omega, u) * (omega[:, d] ** 2)
    assert_close(phi_derivative2, phi_derivative2_manual)

def test_feature_map_derivatives_loop_equals_map_derivative_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives_loop(X, omega, u)
    
    for d in range(D):
        derivative = feature_map_derivative_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives_equals_feature_map_derivatives_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives(X, omega, u)
    derivatives_loop = feature_map_derivatives_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_feature_map_derivatives2_loop_equals_map_derivative2_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives2_loop(X, omega, u)
    
    for d in range(D):
        derivative = feature_map_derivative2_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_feature_map_derivatives2_equals_feature_map_derivatives2_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = feature_map_derivatives2(X, omega, u)
    derivatives_loop = feature_map_derivatives2_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_feature_map_grad_single_equals_feature_map_derivative_d():
    D = 2
    m = 3
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    x = np.random.randn(D)
    
    grad = feature_map_grad_single(x, omega, u)
    
    grad_manual = np.zeros((D, m))
    for d in range(D):
        grad_manual[d, :] = feature_map_derivative_d(x, omega, u, d)
    
    assert_allclose(grad_manual, grad)

def test_compute_b_storage_1d1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    b_manual = -feature_map_derivative2_d(X, omega, u, d).flatten()
    b = compute_b_memory(X, omega, u)
    assert_allclose(b_manual, b)

def test_compute_b_storage_1d2n():
    X = np.array([[1.], [2.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    b_manual = -np.mean(feature_map_derivative2_d(X, omega, u, d))
    b = compute_b_memory(X, omega, u)
    assert_allclose(b_manual, b)

def test_compute_C_1d1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi = feature_map_derivative_d(X, omega, u, d).flatten()
    C_manual = np.outer(phi, phi)
    C = compute_C_memory(X, omega, u)
    assert_allclose(C_manual, C)

def test_compute_C_1d2n():
    X = np.array([[1.], [2.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    C_manual = np.mean(feature_map_derivative_d(X, omega, u, d) ** 2)
    C = compute_C_memory(X, omega, u)
    assert_allclose(C_manual, C)

def test_score_matching_sym():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    lmbda = 1.
    theta = score_matching_sym(X, lmbda, omega, u)
    theta_manual = np.linalg.solve(C + np.eye(m) * lmbda, b)
    assert_allclose(theta, theta_manual)

def test_objective_sym_given_b_C():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    lmbda = 1.
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = np.random.randn(m)
    
    J = objective(X, theta, lmbda, omega, u, b, C)
    J_manual = 0.5 * np.dot(theta.T, np.dot(C + np.eye(m) * lmbda, theta)) - np.dot(theta, b)
    
    assert_close(J, J_manual)

def test_objective_sym_given_b_C_equals_given_nothing():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    lmbda = 1.
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    theta = np.random.randn(m)
    
    J = objective(X, theta, lmbda, omega, u, b, C)
    J2 = objective(X, theta, lmbda, omega, u)
    
    assert_close(J, J2)

def test_objective_sym_equals_completely_manual_manually():
    N = 100
    D = 3
    m = 3
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    lmbda = 1.
    theta = np.random.randn(m)
    
    J_manual = 0.
    for n in range(N):
        b_manual = np.zeros(m)
        C_manual = np.zeros((m, m))
        J_n_manual = 0.
        for d in range(D):
            b_term_manual = -np.sqrt(2. / m) * np.cos(np.dot(X[n], omega) + u) * (omega[d, :] ** 2)
            b_term = feature_map_derivative2_d(X[n], omega, u, d)
            assert_allclose(b_term_manual, b_term)
            b_manual -= b_term_manual
            J_manual += np.dot(b_term_manual, theta)
            J_n_manual += np.dot(b_term_manual, theta)
             
            c_vec_manual = -np.sqrt(2. / m) * np.sin(np.dot(X[n], omega) + u) * omega[d, :]
            c_vec = feature_map_derivative_d(X[n], omega, u, d)
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
        J_n = objective(X[n].reshape(1, m), theta, 0, omega, u)
        J_n_2 = 0.5 * np.dot(theta, np.dot(C, theta)) - np.dot(theta, b)
        assert_allclose(J_n_2, J_n, rtol=1e-4)
        assert_allclose(J_n_manual, J_n, rtol=1e-4)
        
    J_manual /= N
    J_manual += 0.5 * lmbda * np.dot(theta, theta)
    J = objective(X, theta, lmbda, omega, u)

    assert_close(J, J_manual, decimal=5)

def test_objective_sym_equals_completely_manual():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    lmbda = 1.
    theta = np.random.randn(m)
     
    J = objective(X, theta, lmbda, omega, u)
    J_manual = _objective_sym_completely_manual(X, theta, lmbda, omega, u)
     
    assert_close(J_manual, J, decimal=5)

def test_objective_sym_equals_half_manual():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    lmbda = 1.
    theta = np.random.randn(m)
     
    J = objective(X, theta, lmbda, omega, u)
    J_manual = _objective_sym_half_manual(X, theta, lmbda, omega, u)
     
    assert_close(J_manual, J)

# import matplotlib.pyplot as plt
def test_score_matching_sym_returns_min_1d_grid():
    N = 100
    D = 3
    m = 1
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    lmbda = .001
    theta = score_matching_sym(X, lmbda, omega, u)
    J = objective(X, theta, lmbda, omega, u, b, C)
    
    thetas_test = np.linspace(theta - 3, theta + 3)
    Js = np.zeros(len(thetas_test))
    
    for i, theta_test in enumerate(thetas_test):
        Js[i] = objective(X, np.array([theta_test]), lmbda, omega, u, b, C)
    
    
#     plt.plot(thetas_test, Js)
#     plt.plot([theta, theta], [Js.min(), Js.max()])
#     plt.title(str(theta))
#     plt.show()

    assert_almost_equal(Js.min(), J, delta=thetas_test[1] - thetas_test[0])
    assert_almost_equal(thetas_test[Js.argmin()], theta[0], delta=thetas_test[1] - thetas_test[0])

def test_score_matching_sym_returns_min_random_search():
    N = 100
    D = 3
    m = 10
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    X = np.random.randn(N, D)
    
    C = compute_C_memory(X, omega, u)
    b = compute_b_memory(X, omega, u)
    lmbda = 1.
    theta = score_matching_sym(X, lmbda, omega, u)
    J = objective(X, theta, lmbda, omega, u, b, C)
    
    for noise in [0.0001, 0.001, 0.1, 1, 10, 100]:
        for _ in range(10):
            theta_test = np.random.randn(m) * noise + theta
            J_test = objective(X, theta_test, lmbda, omega, u, b, C)
        
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
