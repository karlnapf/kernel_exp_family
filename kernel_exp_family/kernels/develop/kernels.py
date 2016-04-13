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

def compute_all_hessians_old(kernel_dx_dy, data):
    n,d = data.shape
    all_hessians = np.zeros( (n*d, n*d) )

    for a, x_a in enumerate(data):
        for b, x_b in enumerate(data[0:a+1,:]):
            r_start,r_end = a*d, a*d+d
            c_start, c_end = b*d, b*d+d
            all_hessians[r_start:r_end, c_start:c_end] = kernel_dx_dy(x_a.reshape(-1, 1),
                                                                      x_b.reshape(-1, 1))
            all_hessians[c_start:c_end, r_start:r_end] = all_hessians[r_start:r_end, c_start:c_end]

    return all_hessians