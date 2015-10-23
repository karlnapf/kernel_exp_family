from choldate._choldate import cholupdate

import numpy as np


def online_mean_variance(data):
    """
    Computes unbiased estimates for mean and variance for N elements of a given 1d-array.
    Note that the variance is normalised by N-1, unlike the default behaviour of np.var, which normalises by N.
    """
    assert data.ndim == 1
    n = 0
    mean = 0.0
    M2 = 0.0
     
    for x in data:
        n = n + 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + delta * (x - mean)

    if n < 2:
        var = np.nan
    else:
        var = M2 / (n - 1)
    
    return mean, var

def online_mean_covariance(data):
    """
    Computes unbiased estimates for mean and variance for N elements of a given 2d-array of D-dimensional vectors.
    Note that the variance is normalised by N-1, unlike the default behaviour of np.cov, which normalises by N.
    Also note that np.cov takes a transposed X as input compared to this function.
    """
    assert data.ndim == 2
    D = data.shape[1]
    
    n = 0
    mean = np.zeros(D)
    M2 = np.zeros((D, D))
     
    for x in data:
        n = n + 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + np.outer(delta, x - mean)

    if n < 2:
        cov = np.zeros((D, D)) + np.nan
    else:
        cov = M2 / (n - 1)
        
    return mean, cov

def rank_one_update_mean_covariance(x, n=0., mean=None, M2=None):
    """
    Given a mean and sum M2 of n vectors, updates all terms for a new observation x and returns:
    mean, covariance, updated n, updated M2
    """
    assert x.ndim == 1
    D = len(x)
    
    # check if first term
    if n == 0 or mean is None or M2 is None:
        n = 0
        mean = np.zeros(D)
        M2 = np.zeros((D, D))
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert M2.ndim == 2
        assert M2.shape[0] == D
        assert M2.shape[1] == D

    
    # update
    n = n + 1
    delta = x - mean
    mean = mean + delta / n
    M2 = M2 + np.outer(delta, x - mean)
    
    if n < 2:
        cov = np.zeros((D, D)) + np.nan
    else:
        cov = M2 / (n - 1)
    
    return mean, cov, n, M2

def rank_m_update_mean_covariance(X, n=0., mean=None, M2=None, ddof=0):
    """
    Given a mean and sum M2 of n vectors, updates all terms for a m new observations X and returns:
    mean, covariance, updated n, updated M2
    
    Optional parameter is degreed of freedom, which corresponds to the normaliser 1/(N-ddof)
    """
    assert X.ndim == 2
    D = X.shape[1]
    
    # check if first term
    if n == 0 or mean is None or M2 is None:
        n = 0
        mean = np.zeros(D)
        M2 = np.zeros((D, D))
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert M2.shape[0] == D
        assert M2.shape[1] == D
    
    # updates
    for x in X:
        n = n + 1
        delta = x - mean
        mean = mean + delta / n
        M2 = M2 + np.outer(delta, x - mean)
    
    cov = M2 / (n - ddof)
    
    return mean, cov, n, M2

def rank_one_update_mean_covariance_cholesky_naive(x, n=0., mean=None, M2=None, M2_L=None):
    """
    Given a mean and sum M2 of n vectors, updates all terms for a m new observations X and returns:
    mean, Cholesky(covariance), updated n, updated M2
    
    Note: Assumes n>=D, otherwise will covariance might not be positive definite
    
    Expensive version just for development purposes
    """
    assert x.ndim == 1
    D = len(x)
    assert n >= D
    
    # check if first term
    if n == 0 or mean is None or M2 is None or M2_L is None:
        n = 0
        mean = np.zeros(D)
        M2 = np.zeros((D, D))
        M2_L = np.zeros((D, D))
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert M2.ndim == 2
        assert M2.shape[0] == D
        assert M2.shape[1] == D
        assert M2_L.shape[0] == D
        assert M2_L.shape[1] == D
        
    
    # update
    n = n + 1
    delta = x - mean
    mean = mean + delta / n
    M2 = M2 + np.outer(delta, x - mean)
    
    if n < 2:
        M2_L = np.zeros((D, D)) + np.nan
        cov_L = np.zeros((D, D)) + np.nan
    else:
        M2_L = np.linalg.cholesky(M2)
        cov_L = M2_L / np.sqrt((n - 1))
    
    return mean, cov_L, n, M2, M2_L


def rank_one_update_mean_covariance_cholesky_lmbda(u, lmbda=.1, mean=None, cov_L=None, nu2=1., gamma2=None):
    """
    Returns updated mean and Cholesky of sum of outer products following a
    (1-lmbda)*old + lmbda* nu2*uu^T
    rule
    
    Optional: If gamma2 is given, an isotropic term gamma2 * I is added to the uu^T part
    
    where old mean and cov_L=Cholesky(old) (lower Cholesky) are given.
    
    Performs efficient rank-one updates of the Cholesky directly.
    """
    assert lmbda >= 0 and lmbda <= 1
    assert u.ndim == 1
    D = len(u)
    
    # check if first term
    if mean is None or cov_L is None :
        # in that case, zero mean and scaled identity matrix
        mean = np.zeros(D)
        cov_L = np.eye(D) * nu2
    else:
        assert len(mean) == D
        assert mean.ndim == 1
        assert cov_L.ndim == 2
        assert cov_L.shape[0] == D
        assert cov_L.shape[1] == D
    
    # update mean
    updated_mean = (1 - lmbda) * mean + lmbda * u
    
    # update Cholesky: first downscale existing Cholesky
    update_cov_L = np.sqrt(1 - lmbda) * cov_L.T
    
    # rank-one update of the centered new vector
    update_vec = np.sqrt(lmbda) * np.sqrt(nu2) * (u - mean)
    cholupdate(update_cov_L, update_vec)
    
    # optional: add isotropic term if specified, requires looping rank-one updates over
    # all basis vectors e_1, ..., e_D
    if gamma2 is not None:
        e_d = np.zeros(D)
        for d in range(D):
            e_d[:] = 0
            e_d[d] = np.sqrt(gamma2)
            
            # could do a Cholesky update, but this routine does a loop over dimensions
            # where the vector only has one non-zero component
            # That is O(D^2) and therefore not efficient when used in a loop
            cholupdate(update_cov_L, np.sqrt(lmbda) * e_d)
            
            # TODO:
            # in contrast, can do a simplified update when knowing that e_d is sparse
            # manual Cholesky update (only doing the d-th component of algorithm on
            # https://en.wikipedia.org/wiki/Cholesky_decomposition#Rank-one_update
#             # wiki (MB) code:
#             r = sqrt(L(k,k)^2 + x(k)^2);
#             c = r / L(k, k);
#             s = x(k) / L(k, k);
#             L(k, k) = r;
#             L(k+1:n,k) = (L(k+1:n,k) + s*x(k+1:n)) / c;
#             x(k+1:n) = c*x(k+1:n) - s*L(k+1:n,k);

    # since cholupdate works on transposed version
    update_cov_L = update_cov_L.T
    
    # done updating Cholesky
    
    return updated_mean, update_cov_L
