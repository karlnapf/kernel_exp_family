from choldate._choldate import cholupdate

from kernel_exp_family.estimators.finite.gaussian import compute_b
from kernel_exp_family.kernels.kernels import rff_feature_map_grad2_d, \
    rff_feature_map_grad_d, rff_feature_map_grad2, rff_feature_map_grad, \
    rff_feature_map, rff_feature_map_single
import numpy as np


def _objective_sym_completely_manual(X, theta, omega, u):
    N = X.shape[0]
    D = X.shape[1]
    m = len(theta)
    
    J_manual = 0.
     
    for n in range(N):
        for d in range(D):
            b_term_manual = -np.sqrt(2. / m) * np.cos(np.dot(X[n], omega) + u) * (omega[d, :] ** 2)
            J_manual -= -np.dot(b_term_manual, theta)
             
            c_vec_manual = -np.sqrt(2. / m) * np.sin(np.dot(X[n], omega) + u) * omega[d, :]
            C_term_manual = np.outer(c_vec_manual, c_vec_manual)
            J_manual += 0.5 * np.dot(theta, np.dot(C_term_manual, theta))
    
    J_manual /= N
    return J_manual

def _objective_sym_half_manual(X, theta, omega, u):
    N = X.shape[0]
    D = X.shape[1]
    
    J_manual = 0.
     
    for n in range(N):
        for d in range(D):
            b_term = -rff_feature_map_grad2_d(X[n], omega, u, d)
            J_manual -= np.dot(b_term, theta)

            c_vec = rff_feature_map_grad_d(X[n], omega, u, d)
            C_term_manual = np.outer(c_vec, c_vec)
            J_manual += 0.5 * np.dot(theta, np.dot(C_term_manual, theta))
    
    J_manual /= N
    return J_manual

def compute_b_memory(X, omega, u):
    assert len(X.shape) == 2
    Phi2 = rff_feature_map_grad2(X, omega, u)
    return -np.mean(np.sum(Phi2, 0), 0)

def update_b_single(x, b, n, omega, u):
    D = omega.shape[0]
    m = omega.shape[1]
    
    projections_sum = np.zeros(m)
    phi = rff_feature_map_single(x, omega, u)
    for d in range(D):
        # second derivative of feature map
        phi2 = -phi * (omega[d, :] ** 2)
        projections_sum -= phi2
    
    # Knuth's running average
    n += 1
    delta = projections_sum - b
    b += delta / n
    
    return b

def compute_b_weighted(X, omega, u, weights):
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]

    X_weighted = (X.T * weights).T

    projections_sum = np.zeros(m)
    Phi2 = rff_feature_map(X_weighted, omega, u)
    for d in range(D):
        projections_sum += np.sum(-Phi2 * (omega[d, :] ** 2), axis=0)
        
    return -projections_sum / np.sum(weights)

def compute_C_memory(X, omega, u):
    assert len(X.shape) == 2
    Phi2 = rff_feature_map_grad(X, omega, u)
    d = X.shape[1]
    N = X.shape[0]
    m = Phi2.shape[2]
    
#     # bottleneck! loop version is very slow
#     C = np.zeros((m, m))
#     t = time.time()
#     for i in range(N):
#         for ell in range(d):
#             phi2 = Phi2[ell, i]
#             C += np.outer(phi2, phi2)
#     print("loop", time.time()-t)
#      
#     # roughly 5x faster than the above loop
#     t = time.time()
#     Phi2_reshaped = Phi2.reshape(N*d, m)
#     C2=np.einsum('ij,ik->jk', Phi2_reshaped, Phi2_reshaped)
#     print("einsum", time.time()-t)
#
#     # cython implementation, is slowest
#     t = time.time()
#     Phi2_reshaped = Phi2.reshape(N*d, m)
#     C3 = outer_sum_cython(Phi2_reshaped)
#     print("cython", time.time()-t)
    
#     t = time.time()

    # fastest version using multicore: tensordot method
    Phi2_reshaped = Phi2.reshape(N * d, m)
    C4 = np.tensordot(Phi2_reshaped, Phi2_reshaped, [0, 0])
#     print("tensordot", time.time()-t)

    return C4 / N

def update_L_C_naive(x, L_C, n, omega, u):
    D = omega.shape[0]
    assert x.ndim == 1
    assert len(x) == D
    m = 1 if np.isscalar(u) else len(u)
    N = 1
    
    C = np.dot(L_C, L_C.T)
    
    C_new = np.zeros((m, m))
    projection = np.dot(x[np.newaxis, :], omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C_new += np.tensordot(temp, temp, [0, 0])
    
    # Knuth's running average
    n = n + 1
    delta = C_new - C
    C += delta / n
    
    return np.linalg.cholesky(C)

def update_L_C_single(x, L_C, n, omega, u):
    D = omega.shape[0]
    assert x.ndim == 1
    assert len(x) == D
    m = 1 if np.isscalar(u) else len(u)
    N = 1
    
    L_C *= np.sqrt(n)
    
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    projection = np.dot(x[np.newaxis, :], omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        
        # here temp is 1xm, this costs O(m^2)
        cholupdate(L_C, temp[0])
        
    # since cholupdate works on transposed version
    L_C = L_C.T
    
    # since the new C has a 1/(n+1) term in it
    L_C /= np.sqrt(n + 1)
    
    return L_C

def compute_C_weighted(X, omega, u, weights):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    X_weighted = (X.T * weights).T
    
    C = np.zeros((m, m))
    projection = np.dot(X_weighted, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C += np.tensordot(temp, temp, [0, 0])

    return C / np.sum(weights)