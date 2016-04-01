from numpy.ma.testutils import assert_close

from kernel_exp_family.estimators.full.gaussian import SE_dx_i_dx_j, \
    SE_dx_i_dx_i_dx_j, SE, SE_dx, KernelExpFullGaussian

import autograd.numpy as np  # Thinly-wrapped numpy
from autograd import hessian


def test_SE_dx_i_dx_j():
    d = 3
    l = 2.0

    x = np.random.randn(d).reshape(-1, 1)
    y = np.random.randn(d).reshape(-1, 1)

    left_arg_hessian = SE_dx_i_dx_j(x, y, l)
    autograd_left_arg_hessian = hessian(lambda a: SE(a, y, l))
    autograd_computation = np.squeeze(autograd_left_arg_hessian(x))

    assert_close(left_arg_hessian, autograd_computation)


def test_SE_dx_i_dx_i_dx_j():
    d = 3
    l = 2.0

    x = np.random.randn(d).reshape(-1, 1)
    y = np.random.randn(d).reshape(-1, 1)

    derivative_matrix = SE_dx_i_dx_i_dx_j(x, y, l)
    autograd_derivative_matrix = hessian(lambda a: SE_dx(a, y, l))
    autograd_computation = np.squeeze(autograd_derivative_matrix(x))

    d = x.shape[0]
    autograd_result = np.zeros((d, d))

    for i in range(d):
        autograd_result[i, :] = autograd_computation[i, i, :]

    assert_close(derivative_matrix, autograd_result)


def test_grad_execute():
    sigma = 1.
    lmbda = 1.
    N = 10
    D = 2
    est = KernelExpFullGaussian(sigma, lmbda, D, N)

    X = np.random.randn(N, D)
    est.fit(X)

    est.grad(np.random.randn(D))
