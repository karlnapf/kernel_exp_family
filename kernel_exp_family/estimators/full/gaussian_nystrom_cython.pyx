import numpy as np
cimport numpy as np
cimport cython

DTYPE = np.float64
ctypedef np.float64_t DTYPE_FLOAT_t
ctypedef np.int64_t DTYPE_INT_t


@cython.boundscheck(False)
def build_system_nystrom(np.ndarray[DTYPE_FLOAT_t, ndim=2] X, np.float sigma, np.float lmbda, np.ndarray[DTYPE_INT_t, ndim=1] inds):
    """
    This is a "flattened" implementation of build_system_nystrom_modular_slow.
    This means that all function calls are simply copied in this function.
    Completely unreadable, but (maybe) easier to Cythonise, thus a bit faster.
    """
    cdef int N = X.shape[0]
    cdef int D = X.shape[1]
    cdef m = inds.shape[0]
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] term1, term2, term3
    cdef float k, term_a, term_b, term_c, term_d, entry, ridge, differences_i, differences_j, G_sum, G_a_b_i_j, G1, G2
    cdef int a, b, i, j
    
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] h_mat
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] h_vec
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=2] A_mn
    cdef np.ndarray[DTYPE_FLOAT_t, ndim=1] b_vec

    # h = compute_h(X, sigma).reshape(-1)
    h_mat = np.zeros((N, D), dtype=DTYPE)
    for b in range(X.shape[0]):
        for a in range(X.shape[0]):
#             h[b, :] += np.sum(gaussian_kernel_dx_dx_dy(x_a, x_b, sigma), axis=0)
            k = np.exp(-np.sum((X[b] - X[a]) ** 2) / sigma)
            term1 = np.sum(k * np.outer((X[a] - X[b]) ** 2, (X[a] - X[b])) * (2 / sigma) ** 3, axis=0)
            term2 = np.sum(k * 2 * np.diag((X[a] - X[b])) * (2 / sigma) ** 2, axis=0)
            term3 = np.sum(k * np.tile((X[a] - X[b]), [D, 1]) * (2 / sigma) ** 2, axis=0)
            h_mat[b, :] += term1 - term2 - term3
    h_vec = (h_mat / N).reshape(-1)
    
    # xi_norm_2 = compute_xi_norm_2(X, sigma)
    xi_norm_2 = 0.
    for a in range(X.shape[0]):
        for b in range(X.shape[0]):
            # xi_norm_2 += np.sum(gaussian_kernel_dx_dx_dy_dy(x_a, x_b, sigma))
            k = np.exp(-np.sum((X[b] - X[a]) ** 2) / sigma)
            term_a = np.sum(k * np.outer((X[a] - X[b]), (X[a] - X[b])) ** 2 * (2.0 / sigma) ** 4)
            term_b = np.sum(k * 6 * np.diag((X[a] - X[b]) ** 2) * (2.0 / sigma) ** 3)  # diagonal (x-y)
            term_c = np.sum((1 - np.eye(D)) * k * np.tile((X[a] - X[b]), [D, 1]).T ** 2 * (2.0 / sigma) ** 3)  # (x_i-y_i)^2 off-diagonal 
            term_d = np.sum(k * (1 + 2 * np.eye(D, dtype=DTYPE)) * (2.0 / sigma) ** 2)
            xi_norm_2 += term_a - term_b - term_c - term_c + term_d
            
    xi_norm_2 /= N ** 2
    
    A_mn = np.zeros((m + 1, N * D + 1), dtype=DTYPE)
    A_mn[0, 0] = np.dot(h_vec, h_vec) / N + lmbda * xi_norm_2
    
    # for row_idx in range(len(inds)):
    #     for col_idx in range(N * D):
    #         A_mn[1 + row_idx, 1 + col_idx] = compute_lower_right_submatrix_component(X, lmbda, inds[row_idx], col_idx, sigma)
    for row_idx in range(m):
        for col_idx in range(N * D):
            # ind_to_ai
            a, i = row_idx / D, row_idx % D
            b, j = col_idx / D, col_idx % D
            
            # gaussian_kernel_hessian_entry
            k = np.exp(-np.sum((X[a] - X[b]) ** 2) / sigma)
            differences_i = X[b,i] - X[a,i]
            differences_j = X[b,j] - X[a,j]
            ridge = 0.
            if i == j:
                ridge = 2. / sigma
            G_a_b_i_j = k * (ridge - 4 * (differences_i * differences_j) / sigma ** 2)
            
            G_sum = 0.
            for idx_n in range(N):
                for idx_d in range(D):
                    # G1 = gaussian_kernel_hessian_entry(x_a, x_n, i, idx_d, sigma)
                    k = np.exp(-np.sum((X[a] - X[idx_n]) ** 2) / sigma)
                    differences_i = X[idx_n,i] - X[a,i]
                    differences_j = X[idx_n,idx_d] - X[a,idx_d]
                    ridge = 0.
                    if i == idx_d:
                        ridge = 2. / sigma
                    G1 = k * (ridge - 4 * (differences_i * differences_j) / sigma ** 2)
                    
                    # G2 = gaussian_kernel_hessian_entry(x_n, x_b, idx_d, j, sigma)
                    k = np.exp(-np.sum((X[idx_n] - X[b]) ** 2) / sigma)
                    differences_i = X[b,idx_d] - X[idx_n, idx_d]
                    differences_j = X[b,j] - X[idx_n, j]
                    ridge = 0.
                    if idx_d == j:
                        ridge = 2. / sigma
                    G2 = k * (ridge - 4 * (differences_i * differences_j) / sigma ** 2)
                    
                    G_sum += G1 * G2
        
            entry = G_sum / N + lmbda * G_a_b_i_j
            A_mn[1 + row_idx, 1 + col_idx] = entry
    
    # A_mn[0, 1:] = compute_first_row_without_storing(X, h, N, lmbda, sigma)
    for ind1 in range(m):
        a, i = row_idx / D, row_idx % D
        for ind2 in range(N * D):
            b, j = col_idx / D, col_idx % D
            # H = gaussian_kernel_hessian_entry(X[a], X[b], i, j, sigma)
            k = np.exp(-np.sum((X[a] - X[b]) ** 2) / sigma)
            differences_i = X[b,i] - X[a,i]
            differences_j = X[b,j] - X[a,j]
            ridge = 0.
            if i==j:
                ridge = 2./sigma
            H = k*(ridge - 4*(differences_i*differences_j)/sigma**2)
            
            A_mn[0, ind1+1] += h_vec[ind2] * H
    A_mn[0, 1:] /= N
    A_mn[0, 1:] += lmbda * h_vec

    A_mn[1:, 0] = A_mn[0, inds + 1]
    
    # b = compute_RHS(h, xi_norm_2)
    b_vec = np.zeros(h_vec.shape[0] + 1, dtype=DTYPE)
    b_vec[0] = -xi_norm_2
    b_vec[1:] = -h_vec
    
    return A_mn, b