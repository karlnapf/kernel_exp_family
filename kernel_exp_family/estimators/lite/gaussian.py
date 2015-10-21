from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np

def compute_b(X, Y, K_XY, sigma):
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

def compute_C(X, Y, K, sigma):
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

def fit(X, Y, sigma, lmbda, K=None):
        # compute kernel matrix if needed
        if K is None:
            K = gaussian_kernel(X, Y, sigma=sigma)
        
        b = compute_b(X, Y, K, sigma)
        C = compute_C(X, Y, K, sigma)

        # solve regularised linear system
        a = -sigma / 2. * np.linalg.solve(C + (K + np.eye(len(C))) * lmbda,
                                          b)
        
        return a
    
def objective(X, Y, sigma, lmbda, alpha, K=None, K_XY=None, b=None, C=None):
    if K_XY is None:
        K_XY = gaussian_kernel(X, Y, sigma=sigma)
    
    if K is None and lmbda > 0:
        if X is Y:
            K = K_XY
        else:
            K = gaussian_kernel(X, sigma=sigma)
    
    if b is None:
        b = compute_b(X, Y, K_XY, sigma)

    if C is None:
        C = compute_C(X, Y, K_XY, sigma)
    
    
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

