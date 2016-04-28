from nose import SkipTest
from nose.tools import assert_almost_equal
from numpy.ma.testutils import assert_close
from numpy.testing.utils import assert_allclose

from kernel_exp_family.kernels.develop.kernels import rff_feature_map_grad_loop, \
    rff_feature_map_grad2_loop, SE_dx_dy, compute_all_hessians_old, SE_dx_dx_dy, \
    SE_dx_dx, SE_dx_dx_dy_dy, SE_dx_i_dx_j, SE_dx_i_dx_i_dx_j
from kernel_exp_family.kernels.kernels import theano_available, gaussian_kernel, \
    gaussian_kernel_grad, rff_feature_map_single, rff_feature_map, \
    rff_feature_map_grad_d, rff_feature_map_grad2_d, rff_feature_map_grad, \
    rff_feature_map_grad2, rff_feature_map_grad_single, rff_sample_basis, \
    gaussian_kernel_hessian, gaussian_kernel_hessians, gaussian_kernel_dx_dx_dy, \
    gaussian_kernel_dx_dx, gaussian_kernel_dx_dx_dy_dy, gaussian_kernel_dx_i_dx_j, \
    gaussian_kernel_dx_i_dx_i_dx_j, gaussian_kernel_dx_component,\
    gaussian_kernel_dx_dx_component, gaussian_kernel_dx_i_dx_i_dx_j_component,\
    gaussian_kernel_dx_i_dx_j_component, gaussian_kernel_hessian_entry
import numpy as np


if theano_available:
    from kernel_exp_family.kernels.kernels import gaussian_kernel_theano, \
    gaussian_kernel_grad_theano, gaussian_kernel_hessian_theano, \
    gaussian_kernel_third_order_derivative_tensor_theano, \
    rff_feature_map_comp_theano, rff_feature_map_comp_grad_theano, \
    rff_feature_map_comp_hessian_theano, \
    rff_feature_map_comp_third_order_tensor_theano


def test_gaussian_kernel_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_theano(x, y, sigma)

def test_gaussian_kernel_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    k = gaussian_kernel_theano(x, y, sigma)
    k_manual = gaussian_kernel(x[np.newaxis, :], y[np.newaxis, :], sigma)[0, 0]
    
    assert_almost_equal(k, k_manual)

def test_gaussian_kernel_grad_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_grad_theano(x, y, sigma)

def test_gaussian_kernel_grad_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    grad = gaussian_kernel_grad_theano(x, y, sigma)
    grad_manual = gaussian_kernel_grad(x, y[np.newaxis, :], sigma)[0]
    print grad_manual
    print grad
    
    assert_allclose(grad, grad_manual)

def test_gaussian_kernel_hessian_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_hessian_theano(x, y, sigma)

def test_gaussian_kernel_third_order_derivative_tensor_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available")
    
    D = 3
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 2.
    
    gaussian_kernel_third_order_derivative_tensor_theano(x, y, sigma)

def test_rff_feature_map():
    x = 3.
    u = 2.
    omega = 2.
    phi = rff_feature_map_single(x, omega, u)
    phi_manual = np.cos(omega * x + u) * np.sqrt(2.)
    assert_close(phi, phi_manual)

def test_rff_feature_map_single_equals_feature_map():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    phis = rff_feature_map(X, omega, u)
    
    for i, x in enumerate(X):
        phi = rff_feature_map_single(x, omega, u)
        assert_allclose(phis[i], phi)

def test_rff_feature_map_derivative_d_1n():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = rff_feature_map_grad_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_rff_feature_map_derivative_d_2n():
    X = np.array([[1.], [3.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative = rff_feature_map_grad_d(X, omega, u, d)
    phi_derivative_manual = -np.sin(X * omega + u) * omega[:, d] * np.sqrt(2.)
    assert_close(phi_derivative, phi_derivative_manual)

def test_rff_feature_map_derivative2_d():
    X = np.array([[1.]])
    u = np.array([2.])
    omega = np.array([[2.]])
    d = 0
    phi_derivative2 = rff_feature_map_grad2_d(X, omega, u, d)
    phi_derivative2_manual = -rff_feature_map(X, omega, u) * (omega[:, d] ** 2)
    assert_close(phi_derivative2, phi_derivative2_manual)

def test_rff_feature_map_derivatives_loop_equals_map_derivative_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad_loop(X, omega, u)
    
    for d in range(D):
        derivative = rff_feature_map_grad_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_rff_feature_map_derivatives_equals_feature_map_derivatives_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad(X, omega, u)
    derivatives_loop = rff_feature_map_grad_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_rff_feature_map_derivatives2_loop_equals_map_derivative2_d():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad2_loop(X, omega, u)
    
    for d in range(D):
        derivative = rff_feature_map_grad2_d(X, omega, u, d)
        assert_allclose(derivatives[d], derivative)

def test_rff_feature_map_derivatives2_equals_feature_map_derivatives2_loop():
    N = 10
    D = 20
    m = 3
    X = np.random.randn(N, D)
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    derivatives = rff_feature_map_grad2(X, omega, u)
    derivatives_loop = rff_feature_map_grad2_loop(X, omega, u)
    
    assert_allclose(derivatives_loop, derivatives)

def test_rff_feature_map_grad_single_equals_feature_map_derivative_d():
    D = 2
    m = 3
    omega = np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    x = np.random.randn(D)
    
    grad = rff_feature_map_grad_single(x, omega, u)
    
    grad_manual = np.zeros((D, m))
    for d in range(D):
        grad_manual[d, :] = rff_feature_map_grad_d(x, omega, u, d)
    
    assert_allclose(grad_manual, grad)

def test_rff_feature_map_comp_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
    
    D = 2
    x = np.random.randn(D)
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    
    for i in range(m):
        rff_feature_map_comp_theano(x, omega[:, i], u[i])

def test_rff_feature_map_comp_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available.")
    
    D = 2
    x = np.random.randn(D)
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    
    phi_manual = rff_feature_map_single(x, omega, u)
    for i in range(m):
        # phi_manual is a monte carlo average, so have to normalise by np.sqrt(m) here
        phi = rff_feature_map_comp_theano(x, omega[:, i], u[i]) / np.sqrt(m)
        assert_close(phi, phi_manual[i])

def test_rff_feature_map_comp_grad_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
       
    D = 2
    x = np.random.randn(D)
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    
    for i in range(m):
        rff_feature_map_comp_grad_theano(x, omega[:, i], u[i])
 
def test_rff_feature_map_grad_theano_result_equals_manual():
    if not theano_available:
        raise SkipTest("Theano not available.")
      
    D = 2
    x = np.random.randn(D)
    X = x[np.newaxis, :]
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    grad_manual = rff_feature_map_grad(X, omega, u)[:, 0, :]
    
    for i in range(m):
        # phi_manual is a monte carlo average, so have to normalise by np.sqrt(m) here
        grad = rff_feature_map_comp_grad_theano(x, omega[:, i], u[i]) / np.sqrt(m)
        assert_close(grad, grad_manual[:, i])
        
def test_rff_feature_map_hessian_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
       
    D = 2
    x = np.random.randn(D)
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    
    for i in range(m):
        rff_feature_map_comp_hessian_theano(x, omega[:, i], u[i])
 
def test_rff_feature_map_third_order_tensor_theano_execute():
    if not theano_available:
        raise SkipTest("Theano not available.")
       
    D = 2
    x = np.random.randn(D)
    m = 10
    sigma = 1.
    omega, u = rff_sample_basis(D, m, sigma)
    
    for i in range(m):
        rff_feature_map_comp_third_order_tensor_theano(x, omega[:, i], u[i])

def test_gaussian_kernel_hessian_equals_SE_dx_dy():
    D = 2
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    H_new = gaussian_kernel_hessian(x,y,sigma)
    H_old = SE_dx_dy(x.reshape(-1,1), y.reshape(-1,1), np.sqrt(sigma / 2.0))

    assert_close(H_new,H_old)

def test_gaussian_kernel_hessians_equals_old():
    D = 3
    N = 20
    X = np.random.randn(N,D)
    sigma = 0.5

    hessians_new = gaussian_kernel_hessians(X,sigma=sigma)

    kernel_dx_dy = lambda x, y: SE_dx_dy(x, y, l=np.sqrt(sigma / 2.0))
    hessians_old = compute_all_hessians_old(kernel_dx_dy, X)

    assert_close(hessians_new,hessians_old)

def test_gaussian_kernel_hessians_non_symmetric_execute():
    D = 3
    N_x = 20
    N_y = 10

    X = np.random.randn(N_x,D)
    Y = np.random.randn(N_y,D)
    sigma = 0.5

    gaussian_kernel_hessians(X, Y, sigma)

def test_gaussian_kernel_dx_dx_dy_equals_SE_dx_dx_dy():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_dx_dy(x,y,sigma)
    reference = SE_dx_dx_dy(x.reshape(-1, 1), y.reshape(-1, 1), l=np.sqrt(sigma/2.0))

    assert_close(implementation, reference)

def test_gaussian_kernel_dx_dx_equals_SE_dx_dx():
    D = 4
    x = np.random.randn(D)
    Y = np.random.randn(1, D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_dx(x, Y, sigma)
    reference = SE_dx_dx(x.reshape(-1, 1), Y.T, l=np.sqrt(sigma/2.0))

    assert_close(implementation, reference.T)

def test_gaussian_kernel_dx_dx_multiple_ys():
    D = 4
    N = 3
    x = np.random.randn(D)
    Y = np.random.randn(N, D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_dx(x, Y, sigma)
    
    for i in range(N):
        reference = SE_dx_dx(x.reshape(-1, 1), Y[i].reshape(-1,1), l=np.sqrt(sigma/2.0))
        
        assert_close(implementation[i], np.squeeze(reference.T))

def test_gaussian_kernel_dx_dx_dy_dy_equals_SE_dx_dx_dy_dy():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_dx_dy_dy(x,y,sigma)
    reference = SE_dx_dx_dy_dy(x.reshape(-1, 1), y.reshape(-1, 1), l=np.sqrt(sigma/2.0))

    assert_close(implementation, reference)

def test_gaussian_kernel_dx_i_dx_j_equals_SE_dx_i_dx_j():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_i_dx_j(x,y,sigma)
    reference = SE_dx_i_dx_j(x.reshape(-1, 1), y.reshape(-1, 1), l=np.sqrt(sigma/2.0))

    assert_close(implementation, reference)

def test_gaussian_kernel_dx_i_dx_i_dx_j_equals_SE_dx_i_dx_i_dx_j():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    implementation = gaussian_kernel_dx_i_dx_i_dx_j(x,y,sigma)
    reference = SE_dx_i_dx_i_dx_j(x.reshape(-1, 1), y.reshape(-1, 1), l=np.sqrt(sigma/2.0))

    assert_close(implementation, reference)

def test_gaussian_kernel_dx_component_equals_grad():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    grad = gaussian_kernel_grad(x,np.atleast_2d(y), sigma)[0]
    for i in range(D):
        dxi = gaussian_kernel_dx_component(x, y, i, sigma)
        assert_allclose(grad[i], dxi)

def test_gaussian_kernel_dx_dx_component_equals_gaussian_kernel_dx_dx():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    dx_dx = gaussian_kernel_dx_dx(x, np.atleast_2d(y), sigma)[0]
    for i in range(D):
        dxi = gaussian_kernel_dx_dx_component(x, y, i, sigma)
        assert_allclose(dx_dx[i], dxi)
        
def test_gaussian_kernel_dx_i_dx_i_dx_j_component_equals_gaussian_kernel_dx_i_dx_i_dx_j():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    dx_i_dx_i_dx_j = gaussian_kernel_dx_i_dx_i_dx_j(x, y, sigma)
    for i in range(D):
        a = gaussian_kernel_dx_i_dx_i_dx_j_component(x, y, i, sigma)
        assert_allclose(dx_i_dx_i_dx_j[i], a)

def test_gaussian_kernel_dx_i_dx_j_component_equals_gaussian_kernel_dx_i_dx_j():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    dx_i_dx_j = gaussian_kernel_dx_i_dx_j(x, y, sigma)
    for i in range(D):
        a = gaussian_kernel_dx_i_dx_j_component(x, y, i, sigma)
        assert_allclose(dx_i_dx_j[i], a)

def test_gaussian_kernel_hessian_entry_equals_gaussian_kernel_hessian():
    D = 4
    x = np.random.randn(D)
    y = np.random.randn(D)
    sigma = 0.5

    H = gaussian_kernel_hessian(x, y, sigma)
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            h = gaussian_kernel_hessian_entry(x, y, i, j, sigma)
            assert_allclose(H[i,j], h)