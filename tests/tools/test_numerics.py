from nose.tools import assert_equal
from numpy.testing.utils import assert_allclose
import unittest

from kernel_exp_family.tools.numerics import log_sum_exp, log_mean_exp, \
    avg_prob_of_log_probs
import numpy as np


class Test(unittest.TestCase):
    def test_log_sum_exp(self):
        X = np.abs(np.random.randn(100))
        direct = np.log(np.sum(np.exp(X)))
        indirect = log_sum_exp(X)
        assert_allclose(direct, indirect)
        
    def test_log_mean_exp(self):
        X = np.abs(np.random.randn(100))
        direct = np.log(np.mean(np.exp(X)))
        indirect = log_mean_exp(X)
        assert_allclose(direct, indirect)
    
    def test_log_mean_exp_equals_avg_prob_of_log_probs(self):
        X = np.abs(np.random.randn(100))
        direct_exp_log_mean_exp = np.exp(log_mean_exp(X))
        safe = avg_prob_of_log_probs(X)
        assert_allclose(direct_exp_log_mean_exp, safe)
    
    def test_log_mean_exp_fail(self):
        X = np.abs(np.random.randn(100))
        X[0] = -3000
        direct_exp_log_mean_exp = np.exp(log_mean_exp(X))
        assert_equal(direct_exp_log_mean_exp, np.inf)
    
    def test_log_mean_exp_fail_avg_prob_of_log_probs_succ(self):
        X = np.abs(np.random.randn(100))
        X[0] = 3000
        safe = avg_prob_of_log_probs(X)
        
        temp = X.copy()
        temp[0] = 0
        temp = np.exp(temp)
        temp[0] = 0
        manual_safe = np.mean(temp)
        
        assert_allclose(safe, manual_safe)
        
if __name__ == "__main__":
    # import sys;sys.argv = ['', 'Test.testName']
    unittest.main()
