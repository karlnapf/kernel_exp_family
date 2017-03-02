import time

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

def square_dist_mat_graph(X,X2=None):
#     X = X / self.lengthscales
    Xs = tf.reduce_sum(tf.square(X), 1)
    if X2 is None:
        return -2 * tf.matmul(X, tf.transpose(X)) + \
               tf.reshape(Xs, (-1, 1)) + tf.reshape(Xs, (1, -1))
    else:
#         X2 = X2 / self.lengthscales
        X2s = tf.reduce_sum(tf.square(X2), 1)
        return -2 * tf.matmul(X, tf.transpose(X2)) + \
               tf.reshape(Xs, (-1, 1)) + tf.reshape(X2s, (1, -1))
               
def gaussian_kernel_matrix_graph(X,X2=None):
    return tf.exp(-square_dist_mat_graph(X, X2) / tf.exp(log_sigma))

def xi_graph(K_dx_dx):
#     xi = tf.zeros([1])
#     m = X_basis.get_shape().as_list()[0]
#     for a in range(m):
# #         _, p2 = get_first_and_second_partials(k_a, x_a)
#         
#         p2 = K_dx_dx[a]
#         xi += tf.reduce_sum(p2)
#     xi /= m
    
    return tf.reduce_sum(K_dx_dx) / m

def betasum_graph(K_dx):
#     betasum = tf.zeros([1])
#     
#     m = X_basis.get_shape().as_list()[0]
#     for a in range(m):
#         p1 = K_dx[a]
# #         p1, _ = get_first_and_second_partials(gaussian_kernel_graph(x_a,y), x_a)
#         betasum += tf.reduce_sum(tf.mul(p1, beta[a*d : (a+1)*d]))
#     # the kernel derivative has a different sign compared to manual implementation
#     return -betasum
        
    return -tf.reduce_sum(tf.mul(tf.reshape(K_dx, [-1]), beta))
    

def xi_and_betasum_graph(K_dx, K_dx_dx):
    betasum = tf.reduce_sum(tf.mul(tf.reshape(K_dx, [-1]), beta))
    xi = tf.reduce_mean(K_dx_dx)
    
    # the kernel derivative has a different sign compared to manual implementation
    return xi, -betasum

def xi_and_betasum_multiple_graph(K_dx, K_dx_dx):
#     betasum = tf.reduce_sum(tf.mul(tf.reshape(K_dx, [-1]), beta))
    Betasums = []
    n = X.get_shape().as_list()[0]
    for b in range(n):
        Betasums.append(tf.reduce_sum(tf.mul(tf.reshape(K_dx[:,b], [-1]), beta)))
    Betasums= tf.pack(Betasums)
    
#     Betasums = tf.reduce_sum(tf.mul(tf.reshape(K_dx, [-1]), beta))
        
    xi = tf.reduce_mean(K_dx_dx, axis=1)
    
    return xi, Betasums

def f_graph(y):
    K = gaussian_kernel_matrix_graph(X_basis, tf.reshape(y, (-1,1)))
    K_dx = tf.gradients(K, X_basis)[0]
    K_dx_dx = tf.gradients(K_dx, X_basis)[0]
    
    xi, betasum = xi_and_betasum_graph(K_dx, K_dx_dx) 
    return alpha * xi + betasum

def f_graph_multiple(X):
    n = X.get_shape().as_list()[0]
    
    F = []
    for b in range(n):
        F += [f_graph(X[b,:])]
    F = tf.pack(F)
    
    return F

def f_graph_multiple_batch(X):
    n = X.get_shape().as_list()[0]

    K = gaussian_kernel_matrix_graph(X_basis, X)
    K_dx = tf.pack([tf.gradients(K[:,b], X_basis)[0] for b in range(n)])
    K_dx_dx = tf.pack([tf.gradients(K_dx[b], X_basis)[0] for b in range(n)])
    
    # even though this computes the same thing, the resulting score objective is wrong, no idea why
    # for now sticking to the loop below
#     xi, betasum = xi_and_betasum_multiple_graph(K_dx, K_dx_dx)
#     temp = alpha * tf.squeeze(xi) + tf.squeeze(betasum)
#     return alpha * tf.squeeze(xi) + tf.squeeze(betasum)

    F = []
    for b in range(n):
        xi_, betasum_ = xi_and_betasum_graph(K_dx[b], K_dx_dx[b])
        F.append(alpha * xi_ + betasum_)
        
    F = tf.pack(F)
    return F

def score_matching_loss_graph(X):
    
    print "1"
    F = f_graph_multiple_batch(X)
    print "1"
    
    print "2"
    F_dy = tf.gradients(F, X)[0]
    F_dy_dy = tf.gradients(F_dy, X)[0]
    print "2"
    
    print "3"
    scores = tf.reduce_sum(0.5*tf.square(F_dy), axis=1) + tf.reduce_sum(F_dy_dy, axis=1)
    print "3"
    return tf.reduce_mean(scores)
#     n = X.get_shape().as_list()[0]
#     score = tf.zeros([1])
#     for b in range(n):
#         score += tf.reduce_sum(0.5*tf.square(F_dy[b])) + tf.reduce_sum(F_dy_dy[b])
#     return score/n

def score_matching_loss_gradient_graph(X):
    return tf.gradients(score_matching_loss_graph(X), X_basis)[0]


if __name__ == "__main__":
    
    n=3
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
    
    K = gaussian_kernel_matrix_graph(X_basis, tf.reshape(y, (-1,1)))
    K_dx = tf.gradients(K, X_basis)[0]
    K_dx_dx = tf.gradients(K_dx, X_basis)[0]
    
    K_multiple = gaussian_kernel_matrix_graph(X_basis, X)
    K_dx_multiple = tf.pack([tf.gradients(K_multiple[:,b], X_basis)[0] for b in range(n)])
    K_dx_dx_multiple = tf.pack([tf.gradients(K_dx_multiple[b], X_basis)[0] for b in range(n)])
    
    # checked matches go here
    if False:
        print "gaussian_kernel_graph"
        print sess.run(gaussian_kernel_graph(x,y), feed_dict=data)
        print gaussian_kernel(np.atleast_2d(data[x]), np.atleast_2d(data[y]), sigma=sigma)
        
        print "xi_graph"
        print sess.run(xi_graph(K_dx_dx), feed_dict=data)
        print xi_log_pdf(data[y], data[X_basis], sigma, data[alpha])
        
        print "betasum_graph"
        print sess.run(betasum_graph(K_dx), feed_dict=data)
        print betasum_log_pdf(data[y], data[X_basis], sigma, data[beta].reshape(m, d))
        
        print "xi_and_betasum_graph"
        print sess.run(xi_and_betasum_graph(K_dx, K_dx_dx), feed_dict=data)
        print xi_log_pdf(data[y], data[X_basis], sigma, data[beta])
        print betasum_log_pdf(data[y], data[X_basis], sigma, data[beta].reshape(m, d))
        
    if False:
        print "xi_and_betasum_multiple graph"
        print sess.run(xi_and_betasum_multiple_graph(K_dx_multiple, K_dx_dx_multiple), feed_dict=data)
        for i in range(3):
            print sess.run(xi_and_betasum_graph(K_dx_multiple[:,i], K_dx_dx_multiple[:,i]), feed_dict=data)
        
    if False:
        print "f_graph"
        print sess.run(f_graph(y), feed_dict=data)
        print log_pdf(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
    
        print "grad of f_graph"
        print sess.run(tf.gradients(f_graph(y), y), feed_dict=data)
        print grad(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
    
        print "second order grad of f_graph"
        print second_order_grad(data[y], data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
        print sess.run(tf.gradients(tf.gradients(f_graph(y), y)[0], y), feed_dict=data)
    
        print "f_graph multiple batch"
        print sess.run(f_graph_multiple(X), feed_dict=data)
        print sess.run(f_graph_multiple_batch(X), feed_dict=data)
    
    if True:
        print "score_matching_loss_graph"
        g = score_matching_loss_graph(X)
        writer = tf.summary.FileWriter("/home/heiko/tensorflow_logs/", graph=tf.get_default_graph())
        tf.summary.scalar("accuracy", g)
        print "score_matching_loss_graph"
        start = time.time()
        print sess.run(g, feed_dict=data)
        end = time.time()
        print "tf computing took %.4f s" % (end-start)
        
        start = time.time()
        print compute_objective(np.atleast_2d(data[X]), data[X_basis], sigma, data[alpha], data[beta].reshape(m, d))
        end = time.time()
        print "numpy computing took %.4f s" % (end-start)
        
    if False:
        print "score_matching_loss_graph gradients"
        start = time.time()
        g = score_matching_loss_graph(X)
        end = time.time()
        print "graph took %.4f s" % (end-start)
        
        start = time.time()
        g_d = tf.gradients(g, X_basis)
        end = time.time()
        print "gradient took %.4f s" % (end-start)
        
        start = time.time()
        print sess.run(g_d, feed_dict=data)
        end = time.time()
        print "evaluation took %.4f s" % (end-start)
        
    if False:
        print "gaussian_kernel_matrix_graph"
        print sess.run(gaussian_kernel_matrix_graph(X), feed_dict=data)[0,1]
        print sess.run(gaussian_kernel_graph(tf.unpack(X)[0],tf.unpack(X)[1]), feed_dict=data)
    
