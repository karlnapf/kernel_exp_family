from kernel_exp_family.kernels.kernels import rff_feature_map
import numpy as np


def rff_feature_map_grad_loop(X, omega, u):
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

def rff_feature_map_grad2_loop(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections[d, :, :] = -Phi2
        projections[d, :, :] *= omega[d, :] ** 2
        
    return projections

def SE(x, y, l=2):
    # ASSUMES COLUMN VECTORS
    diff = x - y;
    return np.squeeze(np.exp(-np.dot(diff.T, diff) / (2 * l ** 2)))

def SE_dx_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.eye(x.size) / l ** 2
    term2 = SE_tmp * np.dot((x - y), (x - y).T) / l ** 4
    return term1 - term2