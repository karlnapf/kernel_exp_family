from numpy.testing.utils import assert_equal

from kernel_exp_family.tools.xvalidation import XVal
import numpy as np


def test_xval_execute_no_shuffle():
    x = XVal(N=10, num_folds=3, shuffle=False)
    
    for train, test in x:
        print train, test

def test_xval_execute_shuffle():
    x = XVal(N=10, num_folds=3, shuffle=True)
    
    for train, test in x:
        print train, test

def test_xval_result_no_suffle():
    x = XVal(N=10, num_folds=3, shuffle=False)
    
    train, test = x.next()
    assert_equal(train, np.array([3, 4, 5, 6, 7, 8, 9]))
    assert_equal(test, np.array([0, 1, 2]))

    train, test = x.next()
    assert_equal(train, np.array([0, 1, 2, 6, 7, 8, 9]))
    assert_equal(test, np.array([3, 4, 5, ]))
    
    train, test = x.next()
    assert_equal(train, np.array([0, 1, 2, 3, 4, 5]))
    assert_equal(test, np.array([6, 7, 8, 9]))

def test_xval_result_suffle():
    x = XVal(N=10, num_folds=3, shuffle=True)
    
    for train, test in x:
        sorted_all = np.sort(np.hstack((train, test)))
        assert_equal(sorted_all, np.arange(10))
        
        assert len(train) == len(np.unique(train))
        assert len(test) == len(np.unique(test))

        assert np.abs(len(test) - 10 / 3) <= 1
        assert np.abs(len(train) - (10 - 10 / 3)) <= 1
