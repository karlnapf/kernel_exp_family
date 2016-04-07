import numpy as np

from kernel_exp_family.estimators.full.gaussian import compute_G, compute_h




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

def compute_RHS_loop(kernel_dx_dx_dy, data, xi_norm_2):
    n, d = data.shape

    b = np.zeros((n * d + 1, 1))
    b[0] = -xi_norm_2

    h = compute_h(kernel_dx_dx_dy, data)
    for a in range(n):
        for i in range(d):
            b[1 + a * d + i] = -h[a, i]

    return b