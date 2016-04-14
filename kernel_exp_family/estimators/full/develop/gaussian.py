from kernel_exp_family.kernels.develop.kernels import SE_dx_dx_dy, SE_dx_dy,\
    SE_dx_dx_dy_dy
from kernel_exp_family.kernels.kernels import gaussian_kernel_dx_i_dx_i_dx_j,\
    gaussian_kernel_dx_i_dx_j, gaussian_kernel_grad, gaussian_kernel_dx_dx
import numpy as np


def log_pdf_naive(x, X, sigma, alpha, beta):
    N, D = X.shape
    
    xi = 0
    betasum = 0
    for a in range(N):
        x_a = np.atleast_2d(X[a, :])
        gradient_x_xa = np.squeeze(gaussian_kernel_grad(x, x_a, sigma))
        xi_grad = np.squeeze(gaussian_kernel_dx_dx(x, x_a, sigma))
        for i in range(D):
            xi += xi_grad[i] / N
            betasum += gradient_x_xa[i] * beta[a, i]
    
    return alpha * xi + betasum

def grad_naive(x, X, sigma, alpha, beta):
    N, D = X.shape
    
    xi_grad = 0
    betasum_grad = 0
    for a, x_a in enumerate(X):
        xi_gradient_vec = gaussian_kernel_dx_i_dx_i_dx_j(x, x_a, sigma)
        left_arg_hessian = gaussian_kernel_dx_i_dx_j(x, x_a, sigma)
        
        for i in range(D):
            xi_grad += xi_gradient_vec[i] / N
            betasum_grad += beta[a, i] * left_arg_hessian[i]

    return alpha * xi_grad + betasum_grad

def compute_lower_right_submatrix_loop(kernel_dx_dy, data, lmbda):
    n, d = data.shape
    G = compute_G(kernel_dx_dy, data)

    A = np.zeros( (n*d, n*d) )
    for a in range(n):
        for i in range(d):
            for b in range(n):
                for j in range(d):
                    A[b * d + j, a * d + i] = np.sum(G[a, :, i, :] * G[:, b, :, j]) / n + lmbda * G[a, b, i, j]

    return A


def compute_G(kernel_dx_dy, data):
    n, d = data.shape
    G = np.zeros((n, n, d, d))
    for a in range(n):
        for b in range(n):
            x = data[a, :].reshape(-1, 1)
            y = data[b, :].reshape(-1, 1)
            G[a, b, :, :] = kernel_dx_dy(x, y)

    return G


def compute_RHS_loop(kernel_dx_dx_dy, data, xi_norm_2):
    n, d = data.shape

    b = np.zeros((n * d + 1, 1))
    b[0] = -xi_norm_2

    h = compute_h_old_interface(kernel_dx_dx_dy, data)
    for a in range(n):
        for i in range(d):
            b[1 + a * d + i] = -h[a, i]

    return b

def build_system_loop(X, sigma, lmbda):
        # esben parametrised kernel in terms of l as exp(-||---|| / (2*l^2)
    # therefore sigma = 2*(l**2)
    l = np.sqrt(np.float(sigma) / 2)

    n, d = X.shape

    SE_dx_dx_dy_l = lambda x, y: SE_dx_dx_dy(x, y, l)
    SE_dx_dy_l = lambda x, y: SE_dx_dy(x, y, l)
    SE_dx_dx_dy_dy_l = lambda x, y: SE_dx_dx_dy_dy(x, y, l)

    h = compute_h_old_interface(SE_dx_dx_dy_l, X)
    G = compute_G(SE_dx_dy_l, X)
    xi_norm_2 = compute_xi_norm_2_old_interface(SE_dx_dx_dy_dy_l, X)

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

    return A, b


def compute_h_old_interface(kernel_dx_dx_dy, data):
    n, d = data.shape
    h = np.zeros((n, d))
    for b in range(n):
        for a in range(n):
            h[b, :] += np.sum(kernel_dx_dx_dy(data[a, :].reshape(-1, 1), data[b, :].reshape(-1, 1)), axis=0)
            
    return h / n

def compute_xi_norm_2_old_interface(kernel_dx_dx_dy_dy, data):
    n, _ = data.shape
    norm_2 = 0
    for a in range(n):
        for b in range(n):
            x = data[a, :].reshape(-1, 1)
            y = data[b, :].reshape(-1, 1)
            norm_2 += np.sum(kernel_dx_dx_dy_dy(x, y))
            
    return norm_2 / n ** 2