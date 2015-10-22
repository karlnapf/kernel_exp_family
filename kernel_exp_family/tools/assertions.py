import numpy as np

def assert_array_shape(a, ndim=None, shape=None, dims={}):
    if not type(a) is np.ndarray:
        raise TypeError("Provided object type (%s) is not nunpy.array." % str(type(a)))
    
    if ndim is not None:
        if not a.ndim == ndim:
            raise ValueError("Provided array dimensions (%d) are not as expected (%d)." % (a.ndim, ndim))
    
    if shape is not None:
        if not np.all(a.shape==shape):
            raise ValueError("Provided array size (%s) are not as expected (%s)." % (str(a.shape), shape))
    
    for k,v in dims.items():
        if not a.shape[k] == v:
            raise ValueError("Provided array's %d-th dimension's size (%d) is not as expected (%d)." % (k, a.shape[k], v))

def assert_positive_int(i):
    if not type(i) is np.int:
        raise TypeError("Provided argument (%s) must be npumpy.int." % str(type(i)))
    
    if not i>0:
        raise ValueError("Provided integer (%d) must be positive." % i)