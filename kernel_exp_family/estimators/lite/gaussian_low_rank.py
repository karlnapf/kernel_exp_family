from scipy.sparse.linalg.interface import LinearOperator
from scipy.sparse.linalg.isolve.iterative import bicgstab

import numpy as np

def compute_b(X, Y, L_X, L_Y, sigma):
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

def apply_left_C(v, X, Y, L_X, L_Y, lmbda):
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

def fit(X, Y, sigma, lmbda, L_X, L_Y,
                                                    cg_tol=1e-3,
                                                    cg_maxiter=None):
        if cg_maxiter is None:
            # CG needs at max dimension many iterations
            cg_maxiter = L_X.shape[0]
        
        NX = X.shape[0]
        
        # set up and solve regularised linear system via bicgstab
        # this never stores an NxN matrix
        b = compute_b(X, Y, L_X, L_Y, sigma)
        matvec = lambda v:apply_left_C(v, X, Y, L_X, L_Y, lmbda)
        C_operator = LinearOperator((NX, NX), matvec=matvec, dtype=np.float64)
        solution, info = bicgstab(C_operator, b, tol=cg_tol, maxiter=cg_maxiter)
        if info > 0:
            print "Warning: CG not terminated within specified %d iterations" % cg_maxiter
        a = -sigma / 2. * solution
        
        return a

def objective(X, Y, sigma, lmbda, alpha, L_X, L_Y, b=None):
    if b is None:
        b = compute_b()

    N_X = len(X)
    first = 2. / (N_X * sigma) * alpha.dot(b)
    second = 2. / (N_X * sigma ** 2) * alpha.dot(apply_left_C(alpha, X, Y, L_X, L_Y, lmbda))
    J = first + second
    return J
