from scipy.spatial.distance import squareform, pdist, cdist

import numpy as np

try:
    from theano import function
    from theano import tensor as T
    import theano
    theano_available = True
except ImportError:
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

def gaussian_kernel(X, Y=None, sigma=1.):
    assert(len(X.shape) == 2)
    
    # if X==Y, use more efficient pdist call which exploits symmetry
    if Y is None:
        sq_dists = squareform(pdist(X, 'sqeuclidean'))
    else:
        assert(len(Y.shape) == 2)
        assert(X.shape[1] == Y.shape[1])
        sq_dists = cdist(X, Y, 'sqeuclidean')
        
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

def rff_sample_basis(D, m, sigma):
    # rbf sampler is parametrised in gamma, which is at the same time
    # k(x,y) = \exp(-\gamma ||x-y||) and the standard deviation of the spectral density
    gamma = 1./sigma
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