from kernel_exp_family.estimators.full.gaussian import compute_xi_norm_2,\
    xi_log_pdf, log_pdf, betasum_log_pdf, grad, second_order_grad,\
    compute_objective
from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np
import tensorflow as tf


def replace_none_with_zero(l):
    return [0 if i==None else i for i in l] 

def partial(expr, vars, i):
    return tf.gradients(expr, vars)[0][i]

def partial2(expr, vars, i):
    temp = tf.gradients(expr, vars)[0][i]
    return tf.gradients(temp, vars)[0][i]

def dx(expr, vars):
    return tf.pack([partial(expr, vars, i) for i in range(d)])

def dx2(expr, vars):
    return tf.pack([partial2(expr, vars, i) for i in range(d)])

def get_first_and_second_partials(expr, vars):
    grad = tf.gradients(expr, vars)[0]
    grad2 = tf.pack([tf.gradients(grad[i], vars)[0][i] for i in range(d)])
    return grad, grad2

def gaussian_kernel_graph(x,y):
    return tf.exp(-tf.reduce_sum(tf.square(x-y))/tf.exp(log_sigma))

def xi_graph(y):
    xi = tf.zeros([1])
    
    m = X_basis.get_shape().as_list()[0]
    for a in range(m):
        x_a = tf.unpack(X_basis)[a]
        _, p2 = get_first_and_second_partials(gaussian_kernel_graph(x_a,y), x_a)
        xi += tf.reduce_sum(p2)
    
    xi /= m
    
    return xi

def betasum_graph(y):
    betasum = tf.zeros([1])
    
    m = X_basis.get_shape().as_list()[0]
    for a in range(m):
        x_a = tf.unpack(X_basis)[a]
        p1, _ = get_first_and_second_partials(gaussian_kernel_graph(x_a,y), x_a)
        betasum += tf.reduce_sum(tf.mul(p1, beta[a*d : (a+1)*d]))
    
    # the kernel derivative has a different sign compared to manual implementation
    return -betasum

def xi_and_betasum_graph(y):
    betasum = tf.zeros([1])
    xi = tf.zeros([1])
    
    m = X_basis.get_shape().as_list()[0]
    for a in range(m):
        x_a = tf.unpack(X_basis)[a]
        p1, p2 = get_first_and_second_partials(gaussian_kernel_graph(x_a,y), x_a)
        xi += tf.reduce_sum(p2)
        betasum += tf.reduce_sum(tf.mul(p1, beta[a*d : (a+1)*d]))
    
    # the kernel derivative has a different sign compared to manual implementation
    return xi, -betasum

def f_graph(y):
    return alpha * xi_graph(y) + betasum_graph(y)

def score_matching_loss_graph(X):
    score = tf.zeros([1])
    
    n = X.get_shape().as_list()[0]
    for b in range(n):
        x_b = tf.unpack(X)[b]
        f_b = f_graph(x_b)
        
        f_b_dy, f_b_dy2 = get_first_and_second_partials(f_b, x_b)
        
        score += tf.reduce_sum(0.5*tf.square(f_b_dy)) + tf.reduce_sum(f_b_dy2)
    
    return score/n

def score_matching_loss_gradient_graph(X):
    return tf.gradients(score_matching_loss_graph(X), X_basis)[0]


if __name__ == "__main__":
    n=10
    d=1
    m=n
    
    X_init = np.random.randn(n,d)
    lmbda = .01 * 1./n
    sigma = 2.
    
    alpha_beta = np.random.randn(m*d+1)
    
    X = tf.placeholder(name="X", dtype=tf.float32, shape=(n, d))
    X_basis = tf.placeholder(name="X_basis", dtype=tf.float32, shape=(m, d))
    
    beta = tf.placeholder(name="beta", dtype=tf.float32, shape=(m*d))
    alpha = tf.placeholder(name="alpha", dtype=tf.float32)
    
    x = tf.placeholder(name="x", dtype=tf.float32, shape=(d))
    y = tf.placeholder(name="y", dtype=tf.float32, shape=(d))
    log_sigma = tf.placeholder(name="log_sigma", dtype=tf.float32)
    
    
    data={
        X: X_init,
        log_sigma: np.log(sigma),
        X_basis: X_init,
        alpha: alpha_beta[0],
        beta: alpha_beta[1:],
    }
    
    num_threads = 1
    sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=num_threads))
    sess.run(tf.global_variables_initializer())
    
    data[x] = np.zeros(d)
    data[y] = np.ones(d)
    
    # checked matches go here
    if False:
        print "gaussian_kernel_graph"
        print sess.run(gaussian_kernel_graph(x,y), feed_dict=data)
        print gaussian_kernel(np.atleast_2d(data[x]), np.atleast_2d(data[y]), sigma=sigma)
        
        print "xi_graph"
        print sess.run(xi_graph(y), feed_dict=data)
        print xi_log_pdf(data[y], data[X_basis], sigma, data[alpha])
        
        print "betasum_graph"
        print sess.run(betasum_graph(y), feed_dict=data)
        print betasum_log_pdf(data[y], data[X_basis], sigma, data[beta].reshape(m, d))
        
        print "xi_and_betasum_graph"
        print sess.run(xi_and_betasum_graph(y), feed_dict=data)
        print xi_log_pdf(data[y], data[X_basis], sigma, data[alpha], data[beta])
        print betasum_log_pdf(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
    
        print "f_graph"
        print sess.run(f_graph(y), feed_dict=data)
        print log_pdf(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
    
        print "grad of f_graph"
        print sess.run(tf.gradients(f_graph(y), y), feed_dict=data)
        print grad(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
    
        print "second order grad of f_graph"
        print second_order_grad(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
        print sess.run(tf.gradients(tf.gradients(f_graph(y), y)[0], y), feed_dict=data)
    
    print "score_matching_loss_graph"
    print sess.run(score_matching_loss_graph(X), feed_dict=data)
    print compute_objective(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
