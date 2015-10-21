from nose.tools import assert_raises

from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
import numpy as np


def get_instace_KernelExpFiniteGaussian():
    gamma = 2.
    lmbda = 1.
    m = 10
    D = 2
    return KernelExpFiniteGaussian(gamma, lmbda, m, D)

def get_estimator_instances():
    return [
            get_instace_KernelExpFiniteGaussian()
            ]

def test_fit_execute():
    N = 100
    
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)

def test_fit_result_none():
    N = 100
    
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        result = est.fit(X)
        assert result is None

def test_fit_wrong_input_type():
    Xs = [None, "test", 1]
    
    estimators = get_estimator_instances()
    
    for X in Xs:
        for est in estimators:
            assert_raises(TypeError, est.fit, X)

def test_fit_wrong_input_shape():
    N = 100
    
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D, 2)
        assert_raises(ValueError, est.fit, X)

def test_fit_wrong_input_dim():
    N = 100
    
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.fit, X)

def test_log_pdf_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        est.log_pdf(X)

def test_log_pdf_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        result = est.log_pdf(X)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == len(X)

def test_log_pdf_wrong_before_fit():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.log_pdf, X)

def test_log_pdf_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        assert_raises(TypeError, est.log_pdf, None)

def test_log_pdf_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, Y)

def test_log_pdf_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, Y)

def test_log_pdf_single_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.log_pdf_single(x)

def test_log_pdf_single_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.log_pdf_single(x)
        
        assert type(result) is np.float64

def test_log_pdf_single_wrong_before_fit():
    estimators = get_estimator_instances()
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.log_pdf_single, x)

def test_log_pdf_single_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.fit(X)
        assert_raises(TypeError, est.log_pdf_single, None)

def test_log_pdf_single_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_single, x)

def test_log_pdf_single_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_single, x)

def test_grad_single_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.grad_single(x)

def test_grad_single_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.grad_single(x)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == est.D

def test_grad_single_wrong_before_fit():
    estimators = get_estimator_instances()
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.grad_single, x)

def test_grad_single_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.fit(X)
        assert_raises(TypeError, est.grad_single, None)

def test_grad_single_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad_single, x)

def test_grad_single_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad_single, x)

def test_objective_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        est.objective(X)

def test_objective_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        result = est.objective(X)
        
        assert type(result) is np.float64

def test_objective_wrong_before_fit():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.objective, X)

def test_objective_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        assert_raises(TypeError, est.objective, None)

def test_objective_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.objective, Y)

def test_objective_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.objective, Y)