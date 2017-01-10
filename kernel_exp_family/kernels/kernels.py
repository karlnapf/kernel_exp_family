import scipy.spatial.distance

import numpy as np

#########################
# THEANO IMPLEMENTATION #
#########################
try:
    from theano import function
    from theano import tensor as T
    import theano
    theano_available = True
except ImportError:
    theano_available = False

theano_available = False

if theano_available:
    def get_expr_gaussian_kernel(x, y, sigma):
        return T.exp(-((x - y).norm(2) ** 2) / sigma)
    
    def get_expr_gaussian_kernel_grad(x, y, sigma):
        return T.grad(get_expr_gaussian_kernel(x, y, sigma), x)
    
    def get_expr_gaussian_kernel_hessian(x, y, sigma):
        return T.hessian(get_expr_gaussian_kernel(x, y, sigma), x)
    
    def get_expr_gaussian_kernel_third_order_tensor(x, y, sigma):
        grad = get_expr_gaussian_kernel_grad(x, y, sigma)
        G3, updates = theano.scan(lambda i, grad, x: T.hessian(grad[i], x),
                                  sequences=T.arange(grad.shape[0]),
                                  non_sequences=[grad, x])
        return G3, updates
    
    def get_expr_rff_feature_map_component(x, omega, u):
        phi = T.cos(T.dot(x, omega) + u) * T.sqrt(2.)
        return phi
    
    def get_expr_rff_feature_map_component_grad(x, omega, u):
        expr = get_expr_rff_feature_map_component(x, omega, u)
        return T.grad(expr, x)

    def get_expr_rff_feature_map_component_hessian(x, omega, u):
        expr = get_expr_rff_feature_map_component(x, omega, u)
        return T.hessian(expr, x)
    
    def get_expr_rff_feature_map_component_third_order_tensor(x, omega, u):
        grad = get_expr_rff_feature_map_component_grad(x, omega, u)
        G3, updates = theano.scan(lambda i, grad, x: T.hessian(grad[i], x),
                                  sequences=T.arange(grad.shape[0]),
                                  non_sequences=[grad, x])
        return G3, updates
    
    # theano variables
    x = T.dvector('x')
    y = T.dvector('y')
    sigma = T.dscalar('sigma')
    
    # compile function handles
    gaussian_kernel_theano = function(inputs=[x, y, sigma], outputs=get_expr_gaussian_kernel(x, y, sigma))
    gaussian_kernel_grad_theano = function(inputs=[x, y, sigma], outputs=get_expr_gaussian_kernel_grad(x, y, sigma))
    gaussian_kernel_hessian_theano = function(inputs=[x, y, sigma], outputs=get_expr_gaussian_kernel_hessian(x, y, sigma))
    
    G3, updates = get_expr_gaussian_kernel_third_order_tensor(x, y, sigma)
    gaussian_kernel_third_order_derivative_tensor_theano = function([x, y, sigma], G3, updates=updates)

    omega = T.dvector('omega')
    u = T.dscalar('u')
    rff_feature_map_comp_theano = function(inputs=[x, omega, u], outputs=get_expr_rff_feature_map_component(x, omega, u))
    rff_feature_map_comp_grad_theano = function(inputs=[x, omega, u], outputs=get_expr_rff_feature_map_component_grad(x, omega, u))
    rff_feature_map_comp_hessian_theano = function(inputs=[x, omega, u], outputs=get_expr_rff_feature_map_component_hessian(x, omega, u))

    G3, updates = get_expr_rff_feature_map_component_third_order_tensor(x, omega, u)
    rff_feature_map_comp_third_order_tensor_theano = function([x, omega, u], G3, updates=updates)
    
#########################
# MANUAL IMPLEMENTATION #
#########################

def gaussian_kernel(X, Y=None, sigma=1.):
    assert(len(X.shape) == 2)
    
    # if X==Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = scipy.spatial.distance.squareform(scipy.spatial.distance.pdist(X, 'sqeuclidean'))
    else:
        assert(len(Y.shape) == 2)
        assert(X.shape[1] == Y.shape[1])
        sq_dists = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')
        
    K = np.exp(-(sq_dists) / sigma)
    return K

def gaussian_kernel_grad(x, Y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(Y.shape) == 2)
    assert(len(x) == Y.shape[1])
    
    x_2d = x[np.newaxis, :]
    k = gaussian_kernel(x_2d, Y, sigma)
    differences = Y - x
    G = (2.0 / sigma) * (k.T * differences)
    return G

def gaussian_kernel_dx_component(x, y, ell, sigma=1.):
    """
    Ell'th partial derivative wrt left argument
    """
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    assert(len(x) == len(y))
    
    k = gaussian_kernel(np.atleast_2d(x), np.atleast_2d(y), sigma)[0,0]
    difference = y[ell] - x[ell]
    G = (2.0 / sigma) * (k * difference)
    return G

def gaussian_kernel_hessian(x, y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)
    differences = y-x
    H = k*(2*np.eye(d)/sigma - 4*np.outer(differences, differences)/sigma**2)
    return H

def gaussian_kernel_hessian_entry(x, y, i, j, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)[0]
    differences_i = y[i] - x[i]
    differences_j = y[j] - x[j]
    
    ridge = 0.
    if i==j:
        ridge = 2./sigma
    H = k*(ridge - 4*(differences_i*differences_j)/sigma**2)
    return H

def gaussian_kernel_hessians(X, Y=None, sigma=1.0):
    assert(len(X.shape) == 2)

    N_x, d = X.shape
    all_hessians = None

    if Y is None:
        all_hessians = np.zeros((N_x * d, N_x * d))

        for a, x_a in enumerate(X):
            for b, x_b in enumerate(X[0:a + 1, :]):
                r_start, r_end = a * d, a * d + d
                c_start, c_end = b * d, b * d + d
                all_hessians[r_start:r_end, c_start:c_end] = gaussian_kernel_hessian(x_a, x_b, sigma)
                all_hessians[c_start:c_end, r_start:r_end] = all_hessians[r_start:r_end, c_start:c_end]

    else:
        assert(len(Y.shape) == 2)
        assert(X.shape[1] == Y.shape[1])

        N_y = Y.shape[0]
        all_hessians = np.zeros( (N_x*d, N_y*d) )

        for a, x_a in enumerate(X):
            for b, y_b in enumerate(Y):
                all_hessians[a*d:a*d+d, b*d:b*d+d] = gaussian_kernel_hessian(x_a, y_b, sigma)

    return all_hessians


def gaussian_kernel_dx_dx(x, Y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(Y.shape) == 2)
    assert(len(x) == Y.shape[1])

    x_2d = x[np.newaxis, :]
    k = gaussian_kernel(x_2d, Y, sigma)
    sq_differences = (Y - x)**2
    return k.T * (sq_differences*(2.0 / sigma)**2 - 2.0/sigma)

def gaussian_kernel_dx_dx_component(x, y, ell, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    assert(len(x) == len(y))

    k = gaussian_kernel(np.atleast_2d(x), np.atleast_2d(y), sigma)[0,0]
    sq_difference = (y[ell] - x[ell])**2
    return k.T * (sq_difference*(2.0 / sigma)**2 - 2.0/sigma)

def gaussian_kernel_dx_dx_dy(x, y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)
    term1 = k * np.outer((x - y) ** 2, (x - y)) * (2/sigma)**3
    term2 = k * 2 * np.diag((x - y)) * (2/sigma)**2
    term3 = k * np.tile((x - y), [d,1]) * (2/sigma)**2
    return term1 - term2 - term3


def gaussian_kernel_dx_dx_dy_dy(x, y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    
    k = gaussian_kernel(x_2d, y_2d, sigma)
    term1 = k * np.outer((x - y), (x - y)) ** 2 * (2.0/sigma)**4
    term2 = k * 6 * np.diag((x - y) ** 2) * (2.0/sigma)**3  # diagonal (x-y)
    term3 = (1 - np.eye(d)) * k * np.tile((x - y), [d, 1]).T ** 2 * (2.0/sigma)**3  # (x_i-y_i)^2 off-diagonal 
    term5 = k * (1 + 2 * np.eye(d)) * (2.0/sigma)**2
    
    return term1 - term2 - term3 - term3.T + term5


def gaussian_kernel_dx_i_dx_j(x, y, sigma=1.):
    """ Matrix of \frac{\partial k}{\partial x_i \partial x_j}"""
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    pairwise_dist = np.outer(y-x, y-x)

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]

    k = gaussian_kernel(x_2d, y_2d, sigma)
    term1 = k*pairwise_dist * (2.0/sigma)**2
    term2 = k*np.eye(d) * (2.0/sigma)

    return term1 - term2

def gaussian_kernel_dx_i_dx_j_component(x, y, ell, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    pairwise_dist = np.outer(y-x, y-x)

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]

    k = gaussian_kernel(x_2d, y_2d, sigma)
    term1 = k*pairwise_dist * (2.0/sigma)**2
    term2 = k*np.eye(d) * (2.0/sigma)

    return (term1 - term2)[ell]

def gaussian_kernel_dx_i_dx_i_dx_j(x, y, sigma=1.):
    """ Matrix of \frac{\partial k}{\partial x_i^2 \partial x_j}"""
    assert(len(x.shape) == 1), x
    assert(len(y.shape) == 1)
    d = x.size
    
    pairwise_dist_squared_i = np.outer((y-x)**2, y-x)
    row_repeated_distances = np.tile(y-x, [d,1])

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)

    term1 = k*pairwise_dist_squared_i * (2.0/sigma)**3
    term2 = k*row_repeated_distances * (2.0/sigma)**2
    term3 = term2*2*np.eye(d)

    return term1 - term2 - term3

def gaussian_kernel_dx_i_dx_i_dx_j_component(x, y, ell, sigma=1.):
    assert(len(x.shape) == 1), x
    assert(len(y.shape) == 1)
    d = x.size
    
    pairwise_dist_squared_i = np.outer((y-x)**2, y-x)
    row_repeated_distances = np.tile(y-x, [d,1])

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)

    term1 = k*pairwise_dist_squared_i * (2.0/sigma)**3
    term2 = k*row_repeated_distances * (2.0/sigma)**2
    term3 = term2*2*np.eye(d)

    return (term1 - term2 - term3)[ell]

def gaussian_kernel_dx_i_dx_i_dx_j_dx_j(x,y, sigma=1.):
    assert(len(x.shape) == 1)
    assert(len(y.shape) == 1)
    d = x.size

    pairwise_dist_squared = np.outer((y-x)**2, (y-x)**2)
    row_repeated_squared_distances = np.tile((y-x)**2, [d,1])

    x_2d = x[np.newaxis,:]
    y_2d = y[np.newaxis,:]
    k = gaussian_kernel(x_2d, y_2d, sigma)

    term1 = pairwise_dist_squared * (2.0/sigma)**4
    term2 = row_repeated_squared_distances * (2.0/sigma)**3
    term3 = term2.T
    term4 = np.diag((y-x)**2) * (2.0**5/sigma**3)
    term5 = np.eye(d) * (2.0**3/sigma**2)

    return k*(term1 - term2 - term3 - term4 + term5 + (2.0/sigma)**2)

def rff_sample_basis(D, m, sigma):
    # rbf sampler is parametrised in gamma, which is at the same time
    # k(x,y) = \exp(-\gamma ||x-y||) and the standard deviation of the spectral density
    gamma = 1. / sigma
    omega = gamma * np.random.randn(D, m)
    u = np.random.uniform(0, 2 * np.pi, m)
    
    return omega, u

def rff_feature_map_single(x, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    return np.cos(np.dot(x, omega) + u) * np.sqrt(2. / m)

def rff_feature_map(X, omega, u):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.cos(projection, projection)
    projection *= np.sqrt(2. / m)
    return projection

def rff_feature_map_grad_d(X, omega, u, d):
    m = 1 if np.isscalar(u) else len(u)
    
    projection = np.dot(X, omega) + u
    np.sin(projection, projection)
        
    projection *= omega[d, :]
    projection *= np.sqrt(2. / m)
    return -projection

def rff_feature_map_grad2_d(X, omega, u, d):
    Phi2 = rff_feature_map(X, omega, u)
    Phi2 *= omega[d, :] ** 2
    
    return -Phi2

def rff_feature_map_grad(X, omega, u):
    # equal to the looped version, rff_feature_map_grad_loop
    # TODO make more efficient via vectorising
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

def rff_feature_map_grad2(X, omega, u):
    # equal to the looped version, rff_feature_map_grad2_loop
    # TODO make more efficient via vectorising
    m = 1 if np.isscalar(u) else len(u)
    N = X.shape[0]
    D = X.shape[1]
    
    projections = np.zeros((D, N, m))
    Phi2 = rff_feature_map(X, omega, u)
    for d in range(D):
        projections[d, :, :] = -Phi2
        projections[d, :, :] *= omega[d, :] ** 2
        
    return projections

def rff_feature_map_grad_single(x, omega, u):
    D, m = omega.shape
    grad = np.zeros((D, m))
    
    for d in range(D):
        grad[d, :] = rff_feature_map_grad_d(x, omega, u, d)
    
    return grad
