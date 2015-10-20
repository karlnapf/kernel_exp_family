from sklearn.cross_validation import KFold

import numpy as np


def sample_basis(D, m, gamma):
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    return omega, u

def sample_basis_rational_quadratic(D, m, alpha, beta=1., return_taus=False):
    """
    Given a Gaussian kernel of the form
    k(x,y) = \exp(-\gamma  ||x-y||^2)
           = \exp(-0.5  \tau   ||x-y||^2),
           
    where
    \gamma = 0.5  \tau,
    
    this method returns a random Fourier features basis for an infinite mixture
    of Gaussian kernels (aka rational quadratic kernel)
    k(x,y) = \int d\gamma p(\tau) k(x,y),
    where p(\tau) is a Gamma distribution
    \tau \sim \texttt{Gamma}(\tau | \alpha,\beta), parametrised with
    \alpha - shape parameter
    \beta -  mean parameter (mean=shape*scale = shape/rate)
    
    The parametrisation is such that alpha, beta correspond to the closed form RQ kernel
    k(x,y) = (1+ (||x-y||^2 \tau) / (2 \alpha)),
           = (1+ (||x-y||^2) / (2 \alpha \sigma^2)),
    where \tau = \sigma^2, which is the standard form given in textbooks.
    
    I.e. in the GPML book, Chapter 4.
    """
    
    omega = np.zeros((D, m))
    taus = np.zeros(m)
    
    # sample from mixture of Gaussians
    # where the length scales are distributed according to a Gamma
    for i in range(m):
        # each sample has a different length scale
        #     
        #     mean = shape/rate = shape * scale
        # <=> scale = mean/shape = beta/alpha
        tau = np.random.gamma(shape=alpha, scale=beta / alpha)
        taus[i] = tau
        gamma = 0.5 * tau
        omega[:, i] = gamma * np.random.randn(D)
    
    u = np.random.uniform(0, 2 * np.pi, m)
    
    if return_taus:
        return omega, u, taus
    else:
        return omega, u

def feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2. / m)

def feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2. / m)
    return projection

def feature_map_derivative_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
        
    projection *= omega[d, :]
    projection *= np.sqrt(2. / m)
    return -projection

def feature_map_derivative2_d(X, omega, u, d):
    Phi2 = feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return -Phi2

def feature_map_grad_single(x, omega, u):
    D, m = omega.shape
    grad = np.zeros((D, m))
    
    for d in range(D):
        grad[d, :] = feature_map_derivative_d(x, omega, u, d)
    
    return grad

def feature_map_derivatives_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    for d in range(D):
        projections[d, :, :] = projection
        projections[d, :, :] *= omega[d, :]
    
    projections *= -np.sqrt(2. / m)
    return projections

def feature_map_derivatives2_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    Phi2 = feature_map(X, omega, u)
    for d in range(D):
        projections[d, :, :] = -Phi2
        projections[d, :, :] *= omega[d, :] ** 2
        
    return projections

def feature_map_derivatives(X, omega, u):
    return feature_map_derivatives_loop(X, omega, u)

def feature_map_derivatives2(X, omega, u):
    return feature_map_derivatives2_loop(X, omega, u)

def _objective_sym_completely_manual(X, theta, lmbda, omega, u):
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
    J_manual += 0.5 * lmbda * np.dot(theta, theta)
    return J_manual

def _objective_sym_half_manual(X, theta, lmbda, omega, u):
    N = X.shape[0]
    D = X.shape[1]
    
    J_manual = 0.
     
    for n in range(N):
        for d in range(D):
            b_term = -feature_map_derivative2_d(X[n], omega, u, d)
            J_manual -= np.dot(b_term, theta)

            c_vec = feature_map_derivative_d(X[n], omega, u, d)
            C_term_manual = np.outer(c_vec, c_vec)
            J_manual += 0.5 * np.dot(theta, np.dot(C_term_manual, theta))
    
    J_manual /= N
    J_manual += 0.5 * lmbda * np.dot(theta, theta)
    return J_manual

def compute_b_memory(X, omega, u):
    assert len(X.shape) == 2
    Phi1 = feature_map_derivatives2(X, omega, u)
    return -np.mean(np.sum(Phi1, 0), 0)

def compute_b(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    D = X.shape[1]
    
    projections_sum = np.zeros(m)
    Phi2 = feature_map(X, omega, u)
    for d in range(D):
        projections_sum += np.mean(-Phi2 * (omega[d, :] ** 2), 0)
        
    return -projections_sum

def compute_C_memory(X, omega, u):
    assert len(X.shape) == 2
    Phi2 = feature_map_derivatives(X, omega, u)
    d = X.shape[1]
    N = X.shape[0]
    m = Phi2.shape[2]
    
#     ## bottleneck! use np.einsum
#     C = np.zeros((m, m))
#     t = time.time()
#     for i in range(N):
#         for ell in range(d):
#             phi2 = Phi2[ell, i]
#             C += np.outer(phi2, phi2)
#     print("loop", time.time()-t)
#      
#     #roughly 5x faster than the above loop
#     t = time.time()
#     Phi2_reshaped = Phi2.reshape(N*d, m)
#     C2=np.einsum('ij,ik->jk', Phi2_reshaped, Phi2_reshaped)
#     print("einsum", time.time()-t)
#
#     #cython implementation, is slowest
#     t = time.time()
#     Phi2_reshaped = Phi2.reshape(N*d, m)
#     C3 = outer_sum_cython(Phi2_reshaped)
#     print("cython", time.time()-t)
    
#     t = time.time()
    Phi2_reshaped = Phi2.reshape(N * d, m)
    C4 = np.tensordot(Phi2_reshaped, Phi2_reshaped, [0, 0])
#     print("tensordot", time.time()-t)

    return C4 / N

def compute_C(X, omega, u):
    assert len(X.shape) == 2
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    C = np.zeros((m, m))
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
    projection *= -np.sqrt(2. / m)
    temp = np.zeros((N, m))
    for d in range(D):
        temp = -projection * omega[d, :]
        C += np.tensordot(temp, temp, [0, 0])

    return C / N

def score_matching_sym(X, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
    
    if C is None:
        C = compute_C(X, omega, u)
    
    theta = np.linalg.solve(C + lmbda * np.eye(len(C)), b)
    return theta
    
def objective(X, theta, lmbda, omega, u, b=None, C=None):
    if b is None:
        b = compute_b(X, omega, u)
        
    if C is None:
        C = compute_C(X, omega, u)
    
    I = np.eye(len(theta))
    return 0.5 * np.dot(theta, np.dot(C + lmbda * I, theta)) - np.dot(theta, b)


def xvalidate(Z, lmbda, omega, u, n_folds=5, num_repetitions=1):
    Js = np.zeros((num_repetitions, n_folds))
    
    for j in range(num_repetitions):
        kf = KFold(len(Z), n_folds=n_folds, shuffle=True)
        for i, (train, test) in enumerate(kf):
            # train
            theta = score_matching_sym(Z[train], lmbda, omega, u)
            
            # evaluate
            Js[j, i] = objective(Z[test], theta, lmbda, omega, u)
    
    return np.mean(Js, 0)

