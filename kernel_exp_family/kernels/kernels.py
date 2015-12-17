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

