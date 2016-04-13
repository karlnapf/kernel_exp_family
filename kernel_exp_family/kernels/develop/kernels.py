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


def SE_dx(x, y, l=2):
    return SE(x, y, l) * (y - x) / l ** 2


def SE_dx_dx(x, y, l=2):
    # Doing SE(x,y,l)*((y-x)**2/l**4 - 1/l**2) does not work!
    return SE(x, y, l) * (y - x) ** 2 / l ** 4 - SE(x, y, l) / l ** 2


def SE_dx_dx_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.dot((x - y) ** 2, (x - y).T) / l ** 6
    term2 = SE_tmp * 2 * np.diag((x - y)[:, 0]) / l ** 4
    term3 = SE_tmp * np.repeat((x - y), x.size, 1).T / l ** 4
    return term1 - term2 - term3


def SE_dx_dx_dy_dy(x, y, l=2):
    SE_tmp = SE(x, y, l)
    term1 = SE_tmp * np.dot((x - y), (x - y).T) ** 2 / l ** 8
    term2 = SE_tmp * 6 * np.diagflat((x - y) ** 2) / l ** 6  # diagonal (x-y)
    term3 = (1 - np.eye(x.size)) * SE_tmp * np.repeat((x - y), x.size, 1) ** 2 / l ** 6  # (x_i-y_i)^2 off-diagonal 
    term4 = (1 - np.eye(x.size)) * SE_tmp * np.repeat((x - y).T, x.size, 0) ** 2 / l ** 6  # (x_j-y_j)^2 off-diagonal
    term5 = SE_tmp * (1 + 2 * np.eye(x.size)) / l ** 4
    
    return term1 - term2 - term3 - term4 + term5


def SE_dx_i_dx_j(x, y, l=2):
    """ Matrix of \frac{\partial k}{\partial x_i \partial x_j}"""
    pairwise_dist = (y-x).dot((y-x).T)

    term1 = SE(x, y, l)*pairwise_dist/l**4
    term2 = SE(x, y, l)*np.eye(pairwise_dist.shape[0])/l**2

    return term1 - term2


def SE_dx_i_dx_i_dx_j(x, y, l=2):
    """ Matrix of \frac{\partial k}{\partial x_i^2 \partial x_j}"""
    pairwise_dist_squared_i = ((y-x)**2).dot((y-x).T)
    row_repeated_distances = np.repeat((y-x).T,
                                       pairwise_dist_squared_i.shape[0],
                                       axis=0)

    term1 = SE(x, y, l)*pairwise_dist_squared_i/l**6
    term2 = SE(x, y, l)*row_repeated_distances/l**4
    term3 = term2*2*np.eye(pairwise_dist_squared_i.shape[0])

    return term1 - term2 - term3


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