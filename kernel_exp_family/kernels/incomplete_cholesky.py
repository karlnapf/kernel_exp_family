from kernel_exp_family.kernels.kernels import gaussian_kernel
import numpy as np


def incomplete_cholesky_gaussian(X, sigma, eta, power=1, blocksize=100):
    kernel = lambda X, Y: gaussian_kernel(X, Y, sigma=sigma)
    return incomplete_cholesky(X, kernel, eta, power, blocksize)

def incomplete_cholesky(X, kernel, eta, power=1, blocksize=100):
    """
    Computes the incomplete Cholesky factorisation of the kernel matrix defined
    by samples X and a given kernel. The kernel is evaluated on-the-fly.
    The optional power parameter is used to multiply the kernel output with
    itself.
    
    Original code from "Kernel Methods for Pattern Analysis" by Shawe-Taylor and
    Cristianini.
    Modified to compute kernel on the fly, to use kernels multiplied with 
    themselves (tensor product), and optimised speed via using vector
    operations and not pre-allocate full kernel matrix memory, but rather
    allocate memory of low-rank kernel block-wise
    Changes by Heiko Strathmann
    
    parameters:
    X         - List of input vectors to evaluate kernel on
    kernel    - A kernel function that takes one or two 2d-arrays and computes
                the kernel self- or cross-similarities.
                Returns a psd kernel matrix
    eta       - Precision cutoff parameter for the low-rank approximation.
                If lies is (0,1), where smaller means more accurate, low rank
                dimension is chosen from a residual value
                If lies in [1,\infty), this low-rank dimension is chosen
    power     - Every kernel evaluation is multiplied with itself this number
                of times.
    blocksize - Tuning parameter for speed, determines how rows elements are
                allocated in a block for the (growing) kernel matrix. Larger
                means faster algorithm (to some extend if low rank dimension
                is larger than blocksize)
    
    Output: dictionary with key-value pairs:
    R    - is a low-rank factor such that R.T.dot(R) approximates the
           original K
    K_chol, ell, I, R, W, where
    K    - is the kernel using only the pivot index features
    I    - is a vector containing the pivots used to compute K_chol
    W    - is a matrix such that W.T.dot(K_chol.dot(W)) approximates the
           original K
    nu   - vector of square rooted residuals for the pivoted points
    
    """
    assert(eta > 0)
    assert(power >= 1)
    assert(blocksize >= 1)
    assert(len(X) >= 1)
    
    m = len(X)

    # growing low rank basis
    R = np.zeros((blocksize, m))
    
    # diagonal (assumed to be one)
    d = np.ones(m)
    
    # used indices
    I = []
    nu = []
    
    # algorithm is executed as long as a is bigger than eta precision
    a = d.max()
    I.append(d.argmax())
    
    # growing set of evaluated kernel values
    K = np.zeros((blocksize, m))
    
    j = 0
    
    # Run based on what eta represents
    # If in (0,1), until residuals are smaller than eta
    # If in [1,\infty), until reconstruction has rank of eta
    while eta < 1 and a > eta \
          or eta >= 1 and j < eta:
        nu.append(np.sqrt(a))
        
        if power >= 1:
            K[j, :] = kernel(X[I[j], np.newaxis], X) ** power
        else:
            K[j, :] = 1.
            
        if j == 0:
            R_dot_j = 0
        elif j == 1:
            R_dot_j = R[:j, :] * R[:j, I[j]]
        else:
            R_dot_j = R[:j, :].T.dot(R[:j, I[j]])
                        
        R[j, :] = (K[j, :] - R_dot_j) / nu[j]
        d = d - R[j, :] ** 2
        a = d.max()
        I.append(d.argmax())
        j = j + 1
        
        # allocate more space for kernel
        if j >= len(K):
            K = np.vstack((K, np.zeros((blocksize, m))))
            R = np.vstack((R, np.zeros((blocksize, m))))
            
    # remove un-used rows which were located unnecessarily
    K = K[:j, :]
    R = R[:j, :]

    # remove list pivot index since it is not used
    I = I[:-1]
    
    # from low rank to full rank
    W = np.linalg.solve(R[:, I], R)
    
    # low rank K
    K_chol = K[:, I]
    
    return {"R":R, "K_chol":K_chol, "I":np.asarray(I), "W":W, "nu": np.asarray(nu)}

def incomplete_cholesky_new_point(X, x, kernel, I=None, R=None, nu=None):
    # compute factorisation if needed
    if I is None or R is None:
        temp = incomplete_cholesky(X, kernel, eta=0.8, power=2)
        R, I, nu = (temp["R"], temp["I"], temp["nu"])
    
    # compute kernel between pivot training and all test elements
    k = kernel(x[np.newaxis, :], X[I])[0]
    
    r_new = np.zeros(len(I))
    for j in range(len(r_new)):
        r_new[j] = (k[j] - r_new.dot(R[:, I[j]])) / nu[j]
    
    return r_new

def incomplete_cholesky_new_points(X, X_test, kernel, I=None, R=None, nu=None):
    # compute factorisation if needed
    if I is None or R is None or nu is None:
        temp = incomplete_cholesky(X, kernel, eta=0.8, power=2)
        R, I, nu = (temp["R"], temp["I"], temp["nu"])
    
    # compute kernel between pivot training and all test elements
    ks = kernel(X[I], X_test)
    
    R_new = np.zeros((R.shape[0], len(X_test)))
    for j in range(len(I)):
        R_new[j, :] = (ks[j, :] - R_new.T.dot(R[:, I[j]])) / nu[j]
            
    return R_new
