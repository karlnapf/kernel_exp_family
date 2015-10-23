from nose.tools import assert_raises

from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import numpy as np


def get_instace_KernelExpFiniteGaussian():
    gamma = 2.
    lmbda = 1.
    m = 10
    D = 2
    return KernelExpFiniteGaussian(gamma, lmbda, m, D)

def get_instace_KernelExpLiteGaussian():
    sigma = 2.
    lmbda = 1.
    D = 2
    return KernelExpLiteGaussian(sigma, lmbda, D)

def get_estimator_instances():
    return [
            get_instace_KernelExpFiniteGaussian(),
            get_instace_KernelExpLiteGaussian()
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

def test_log_pdf_multiple_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        est.log_pdf_multiple(X)

def test_log_pdf_multiple_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        result = est.log_pdf_multiple(X)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == len(X)

def test_log_pdf_multiple_wrong_before_fit():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.log_pdf_multiple, X)

def test_log_pdf_multiple_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        assert_raises(TypeError, est.log_pdf_multiple, None)

def test_log_pdf_multiple_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_multiple, Y)

def test_log_pdf_multiple_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_multiple, Y)

def test_log_pdf_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.log_pdf(x)

def test_log_pdf_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.log_pdf(x)
        
        assert type(result) is np.float64

def test_log_pdf_wrong_before_fit():
    estimators = get_estimator_instances()
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.log_pdf, x)

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
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, x)

def test_log_pdf_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, x)

def test_grad_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.grad(x)

def test_grad_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.grad(x)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == est.D

def test_grad_wrong_before_fit():
    estimators = get_estimator_instances()
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            assert_raises(RuntimeError, est.grad, x)

def test_grad_wrong_input_type():
    N = 10
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.fit(X)
        assert_raises(TypeError, est.grad, None)

def test_grad_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad, x)

def test_grad_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad, x)

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

def test_xvalidate_objective_execute():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.xvalidate_objective(X, num_folds=3, num_repetitions=1)

def test_xvalidate_objective_result():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        result = est.xvalidate_objective(X, num_folds=3, num_repetitions=2)
        
        assert type(result) is np.ndarray
        assert result.ndim == 2
        assert result.shape[0] == 2
        assert result.shape[1] == 3

def test_xvalidate_objective_wrong_input_type():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        assert_raises(TypeError, est.xvalidate_objective, X=None, num_folds=3, num_repetitions=2)
        assert_raises(TypeError, est.xvalidate_objective, X=X, num_folds=None, num_repetitions=2)
        assert_raises(TypeError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=None)
        
def test_xvalidate_objective_wrong_input_dim_X():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D, 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=2)

def test_xvalidate_objective_wrong_input_shape_X():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=2)

def test_xvalidate_objective_wrong_input_negative_int():
    N = 100
    estimators = get_estimator_instances()
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=0, num_repetitions=2)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=0)

def test_get_parameters_finite():
    names = get_instace_KernelExpFiniteGaussian().get_parameter_names()
    assert "gamma" in names
    assert "lmbda" in names
    assert len(names) == 2

def test_get_parameters_lite():
    names = get_instace_KernelExpLiteGaussian().get_parameter_names()
    assert "sigma" in names
    assert "lmbda" in names
    assert len(names) == 2

def test_get_parameters():
    estimators = get_estimator_instances()
    
    for estimator in estimators:
        param_dict = estimator.get_parameters()
        for name, value in param_dict.items():
            assert getattr(estimator, name) == value

def test_set_parameters_from_dict():
    estimators = get_estimator_instances()
    
    for estimator in estimators:
        param_dict = estimator.get_parameters()
        param_dict_old = param_dict.copy()
        for name in param_dict.keys():
            param_dict[name] += 1
        
        estimator.set_parameters_from_dict(param_dict)
        
        param_dict_new = estimator.get_parameters()
        for name in param_dict_new.keys():
            assert param_dict_new[name] == param_dict_old[name] + 1
        
def test_set_parameters_from_dict_wrong_input_type():
    estimators = get_estimator_instances()
    
    for estimator in estimators:
        assert_raises(TypeError, estimator.set_parameters_from_dict, None)
        assert_raises(TypeError, estimator.set_parameters_from_dict, 1)
        assert_raises(TypeError, estimator.set_parameters_from_dict, [])
        
def test_set_parameters_from_dict_wrong_input_parameters():
    estimators = get_estimator_instances()
    
    for estimator in estimators:
        param_dict = estimator.get_parameters()
        param_dict['strange_parameter'] = 0
        assert_raises(ValueError, estimator.set_parameters_from_dict, param_dict)
        