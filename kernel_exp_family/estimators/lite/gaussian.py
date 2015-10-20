from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve.iterative import bicgstab

from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np


def _compute_b_sym(Z, K, sigma):
    assert K.shape[0] == Z.shape[0]
    assert K.shape[0] == K.shape[1]
    
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    b = np.zeros(N)
    K1 = np.sum(K, 1)
    for l in np.arange(D):
        x_l = Z[:, l]
        s_l = x_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_K = s_l[:, np.newaxis] * K
        D_x_K = x_l[:, np.newaxis] * K
        
        b += 2. / sigma * (K.dot(s_l) \
                        + np.sum(D_s_K, 1) \
                        - 2 * D_x_K.dot(x_l)) - K1
    return b

def _compute_b(X, Y, K_XY, sigma):
    assert X.shape[1] == Y.shape[1]
    assert K_XY.shape[0] == X.shape[0]
    assert K_XY.shape[1] == Y.shape[0]
    
    NX = len(X)
    D = X.shape[1]
    
    b = np.zeros(NX)
    K1 = np.sum(K_XY, 1)
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        
        s_l = x_l ** 2
        t_l = y_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_K = s_l[:, np.newaxis] * K_XY
        D_x_K = x_l[:, np.newaxis] * K_XY
        
        b += 2. / sigma * (K_XY.dot(t_l) \
                        + np.sum(D_s_K, 1) \
                        - 2 * D_x_K.dot(y_l)) - K1
    
    return b

def _compute_b_low_rank(X, Y, L_X, L_Y, sigma):
    assert X.shape[1] == Y.shape[1]
    assert L_X.shape[0] == X.shape[0]
    assert L_Y.shape[0] == Y.shape[0]
    
    NX = len(X)
    D = X.shape[1]
    
    b = np.zeros(NX)
    LX1 = L_X.dot(np.sum(L_Y.T, 1))
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        s_l = x_l ** 2
        t_l = y_l ** 2
        
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_LX = s_l[:, np.newaxis] * L_X
        D_x_LX = x_l[:, np.newaxis] * L_X
        
        # compute b incrementally exploiting the Cholesky factorisation of K
        b += 2. / sigma * (L_X.dot(L_Y.T.dot(t_l)) \
                        + D_s_LX.dot(np.sum(L_Y.T, 1)) \
                        - 2 * D_x_LX.dot(L_Y.T.dot(y_l))) - LX1
    
    return b

def _compute_b_low_rank_sym(Z, L, sigma):
    assert len(Z) == len(L)
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    b = np.zeros(N)
    L1 = L.dot(np.sum(L.T, 1))
    for l in np.arange(D):
        x_l = Z[:, l]
        s_l = x_l ** 2
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_s_L = s_l[:, np.newaxis] * L
        D_x_L = x_l[:, np.newaxis] * L
        
        # compute b incrementally exploiting the Cholesky factorisation of K
        b += 2. / sigma * (L.dot(L.T.dot(s_l)) \
                        + D_s_L.dot(np.sum(L.T, 1)) \
                        - 2 * D_x_L.dot(L.T.dot(x_l))) - L1
    
    return b

def _compute_C_sym(Z, K, sigma):
    assert K.shape[0] == Z.shape[0]
    assert K.shape[0] == K.shape[1]
    
    D = Z.shape[1]
    N = Z.shape[0]
    
    C = np.zeros((N, N))
    for l in np.arange(D):
        x_l = Z[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_K = x_l[:, np.newaxis] * K
        K_D_x = K * x_l
        
        C += (D_x_K - K_D_x).dot(K_D_x - D_x_K)
    
    return C

def _compute_C(X, Y, K, sigma):
    assert X.shape[1] == Y.shape[1]
    assert K.shape[0] == X.shape[0]
    assert K.shape[1] == Y.shape[0]
    
    D = X.shape[1]
    NX = X.shape[0]
    
    C = np.zeros((NX, NX))
    for l in np.arange(D):
        x_l = X[:, l]
        y_l = Y[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_KXY = x_l[:, np.newaxis] * K
        KXY_D_y = K * y_l
        KXY_T_D_x = K.T * x_l
        D_y_KXY_T = y_l[:, np.newaxis] * K.T
        
        C += (D_x_KXY - KXY_D_y).dot(KXY_T_D_x - D_y_KXY_T)
    
    return C

def _apply_left_C_sym_low_rank(v, Z, L, lmbda):
    assert len(Z) == len(L)
    assert len(v.shape) == 1
    
    N = Z.shape[0]
    D = Z.shape[1]
    
    # multiply C to v (C is a sum over d=1...D)
    result = np.zeros(N)
    for l in range(D):
        x_l = Z[:, l]
        
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
        D_x_L = x_l[:, np.newaxis] * L
        L_T_D_x = L.T * x_l
        
        # right term of product
        x = L.T.dot(v)
        x = D_x_L.dot(x)
        y = L_T_D_x.dot(v)
        y = L.dot(y)
        
        # right term times v
        temp = x - y
        
        # term of product
        x = L_T_D_x.dot(temp)
        x = L.dot(x)
        y = L.T.dot(temp)
        y = D_x_L.dot(y)
        
        # add both terms times v to result
        result += x - y
    
    if lmbda > 0:
        # regularise with K=L_X.dot(L_X.T)
        result += lmbda * L.dot(L.T.dot(v))
        
        # regularise with I
        result += lmbda * v
    
    return result

def _apply_left_C_low_rank(v, X, Y, L_X, L_Y, lmbda):
    assert len(v.shape) == 1
    assert len(X) == len(L_X)
    assert len(Y) == len(L_Y)
    assert L_X.shape[1] == L_Y.shape[1]
    assert X.shape[1] == Y.shape[1]
     
    N_X = X.shape[0]
    D = X.shape[1]
     
    # multiply C to v (C is a sum over d=1...D)
    result = np.zeros(N_X)
    for l in range(D):
        x_l = X[:, l]
        y_l = Y[:, l]
         
        # Replaces dot product with np.diag via broadcasting
        # See http://mail.scipy.org/pipermail/numpy-discussion/2007-March/026809.html
#         D_x_KXY = x_l[:, np.newaxis] * K
#         KXY_D_y = K * y_l
#         KXY_T_D_x = K.T * x_l
#         D_y_KXY_T =  y_l[:, np.newaxis] * K.T
#         C += (D_x_KXY - KXY_D_y).dot(KXY_T_D_x - D_y_KXY_T)
        
        D_x_LX = x_l[:, np.newaxis] * L_X
        LY_T_D_y = L_Y.T * y_l
        LX_T_D_x = L_X.T * x_l
        D_y_LY = y_l[:, np.newaxis] * L_Y
         
        # right term of product
        x = L_X.T.dot(v)
        x = D_y_LY.dot(x)
        y = LX_T_D_x.dot(v)
        y = L_Y.dot(y)
         
        # right term times v
        temp = x - y
         
        # term of product
        x = LY_T_D_y.dot(temp)
        x = L_X.dot(x)
        y = L_Y.T.dot(temp)
        y = D_x_LX.dot(y)
         
        # add both terms times v to result
        result += x - y
     
    if lmbda > 0:
        # regularise with K=L_X.dot(L_X.T)
        result += lmbda * L_X.dot(L_X.T.dot(v))
    
        # regularise with I
        result += lmbda * v
     
    return result

def score_matching_sym(Z, sigma, lmbda, K=None, b=None, C=None):
        # compute quantities
        if K is None:
            K = gaussian_kernel(Z, sigma=sigma)
        
        if b is None:
            b = _compute_b_sym(Z, K, sigma)
        
        if C is None:
            C = _compute_C_sym(Z, K, sigma)

        # solve regularised linear system
        a = -sigma / 2. * np.linalg.solve(C + (K + np.eye(len(C))) * lmbda,
                                          b)
        
        return a

def fit(X, Y, sigma, lmbda, K=None):
        # compute kernel matrix if needed
        if K is None:
            K = gaussian_kernel(X, Y, sigma=sigma)
        
        b = _compute_b(X, Y, K, sigma)
        C = _compute_C(X, Y, K, sigma)

        # solve regularised linear system
        a = -sigma / 2. * np.linalg.solve(C + (K + np.eye(len(C))) * lmbda,
                                          b)
        
        return a
    
def score_matching_sym_low_rank(Z, sigma, lmbda, L,
                                                    cg_tol=1e-3,
                                                    cg_maxiter=None):
        if cg_maxiter is None:
            # CG needs at max dimension many iterations
            cg_maxiter = L.shape[0]
        
        N = Z.shape[0]
        
        # set up and solve regularised linear system via bicgstab
        # this never stores an NxN matrix
        b = _compute_b_low_rank_sym(Z, L, sigma)
        matvec = lambda v:_apply_left_C_sym_low_rank(v, Z, L, lmbda)
        C_operator = LinearOperator((N, N), matvec=matvec, dtype=np.float64)
        solution, info = bicgstab(C_operator, b, tol=cg_tol, maxiter=cg_maxiter)
        if info > 0:
            print "Warning: CG not terminated within specified %d iterations" % cg_maxiter
        a = -sigma / 2. * solution
        
        return a

def score_matching_low_rank(X, Y, sigma, lmbda, L_X, L_Y,
                                                    cg_tol=1e-3,
                                                    cg_maxiter=None):
        if cg_maxiter is None:
            # CG needs at max dimension many iterations
            cg_maxiter = L_X.shape[0]
        
        NX = X.shape[0]
        
        # set up and solve regularised linear system via bicgstab
        # this never stores an NxN matrix
        b = _compute_b_low_rank(X, Y, L_X, L_Y, sigma)
        matvec = lambda v:_apply_left_C_low_rank(v, X, Y, L_X, L_Y, lmbda)
        C_operator = LinearOperator((NX, NX), matvec=matvec, dtype=np.float64)
        solution, info = bicgstab(C_operator, b, tol=cg_tol, maxiter=cg_maxiter)
        if info > 0:
            print "Warning: CG not terminated within specified %d iterations" % cg_maxiter
        a = -sigma / 2. * solution
        
        return a

def _objective_sym(Z, sigma, lmbda, alpha, K=None, b=None, C=None):
    if K is None and ((b is None or C is None) or lmbda > 0):
        K = gaussian_kernel(Z, sigma=sigma)
    
    if C is None:
        C = _compute_C_sym(Z, K, sigma)
    
    if b is None:
        b = _compute_b_sym(Z, K, sigma)
    
    N = len(Z)
    first = 2. / (N * sigma) * alpha.dot(b)
    second = 2. / (N * sigma ** 2) * alpha.dot(
                                               (C + (K + np.eye(len(C))) * lmbda).dot(alpha)
                                               )
    J = first + second
    return J

def _objective(X, Y, sigma, lmbda, alpha, K=None, K_XY=None, b=None, C=None):
    if K_XY is None:
        K_XY = gaussian_kernel(X, Y, sigma=sigma)
    
    if K is None and lmbda > 0:
        if X is Y:
            K = K_XY
        else:
            K = gaussian_kernel(X, sigma=sigma)
    
    if b is None:
        b = _compute_b(X, Y, K_XY, sigma)

    if C is None:
        C = _compute_C(X, Y, K_XY, sigma)
    
    
    NX = len(X)
    first = 2. / (NX * sigma) * alpha.dot(b)
    if lmbda > 0:
        second = 2. / (NX * sigma ** 2) * alpha.dot(
                                                    (C + (K + np.eye(len(C))) * lmbda).dot(alpha)
                                                    )
    else:
        second = 2. / (NX * sigma ** 2) * alpha.dot((C).dot(alpha))
    J = first + second
    return J

def _objective_sym_low_rank(Z, sigma, lmbda, alpha, L, b=None):
    if b is None:
        b = _compute_b_low_rank_sym(Z, L, sigma)
    
    N = len(Z)
    first = 2. / (N * sigma) * alpha.dot(b)
    second = 2. / (N * sigma ** 2) * alpha.dot(
                                               _apply_left_C_sym_low_rank(
                                                                          alpha, Z, L, lmbda
                                                                          )
                                               )
    J = first + second
    return J

def _objective_low_rank(X, Y, sigma, lmbda, alpha, L_X, L_Y, b=None):
    if b is None:
        b = _compute_b_low_rank()

    N_X = len(X)
    first = 2. / (N_X * sigma) * alpha.dot(b)
    second = 2. / (N_X * sigma ** 2) * alpha.dot(_apply_left_C_low_rank(alpha, X, Y, L_X, L_Y, lmbda))
    J = first + second
    return J

def xvalidate(Z, n_folds, sigma, lmbda, K, num_repetitions=1):
    Js = np.zeros((num_repetitions, n_folds))
    
    for j in range(num_repetitions):
        kf = KFold(len(Z), n_folds=n_folds, shuffle=True)
        for i, (train, test) in enumerate(kf):
            # train
            a = score_matching_sym(Z[train], sigma, lmbda, K[train][:, train])
            
            # precompute test statistics
            C = _compute_C(Z[train], Z[test], K[train][:, test], sigma)
            b = _compute_b(Z[train], Z[test], K[train][:, test], sigma)
            
            # evaluate *without* the lambda
            lmbda_equals_0 = 0.
            Js[j, i] = _objective(Z[train], Z[test], sigma, lmbda_equals_0, a,
                               K=K[train][:, train],
                               K_XY=K[train][:, test],
                               b=b,
                               C=C)
    
    return np.mean(Js, 0)

