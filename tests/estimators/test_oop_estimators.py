from nose.tools import assert_raises
from numpy.testing.utils import assert_allclose

from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
import numpy as np


def get_instace_KernelExpFiniteGaussian(N):
    sigma = 2.
    lmbda = 2.
    D = 2
    m = 2
    return KernelExpFiniteGaussian(sigma, lmbda, m, D)

def get_instace_KernelExpLiteGaussian(N):
    sigma = 2.
    lmbda = 1.
    D = 2
    return KernelExpLiteGaussian(sigma, lmbda, D, N)

def get_estimator_instances(N):
    return [
            get_instace_KernelExpFiniteGaussian(N),
            get_instace_KernelExpLiteGaussian(N)
            ]

def test_get_name_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        assert type(est.get_name()) is str

def test_fit_execute():
    N = 100
    
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)

def test_fit_result_none():
    N = 100
    
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        result = est.fit(X)
        assert result is None

def test_fit_wrong_input_type():
    Xs = [None, "test", 1]
    N = 1
    estimators = get_estimator_instances(N)
    
    for X in Xs:
        for est in estimators:
            assert_raises(TypeError, est.fit, X)

def test_fit_wrong_input_shape():
    N = 100
    
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D, 2)
        assert_raises(ValueError, est.fit, X)

def test_fit_wrong_input_dim():
    N = 100
    
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.fit, X)

def test_log_pdf_multiple_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        est.log_pdf_multiple(X)

def test_log_pdf_multiple_result():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        result = est.log_pdf_multiple(X)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == len(X)

def test_log_pdf_multiple_result_before_fit():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        for est in estimators:
            result = est.log_pdf_multiple(X)
            assert_allclose(result, np.zeros(N))

def test_log_pdf_multiple_wrong_input_type():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        assert_raises(TypeError, est.log_pdf_multiple, None)

def test_log_pdf_multiple_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_multiple, Y)

def test_log_pdf_multiple_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf_multiple, Y)

def test_log_pdf_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.log_pdf(x)

def test_log_pdf_result():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.log_pdf(x)
        
        assert type(result) is np.float64

def test_log_pdf_result_before_fit():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            result = est.log_pdf(x)
            assert_allclose(result, 0)

def test_log_pdf_wrong_input_type():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.fit(X)
        assert_raises(TypeError, est.log_pdf, None)

def test_log_pdf_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, x)

def test_log_pdf_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.log_pdf, x)

def test_grad_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        est.grad(x)

def test_grad_result():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D)
        est.fit(X)
        result = est.grad(x)
        
        assert type(result) is np.ndarray
        assert result.ndim == 1
        assert len(result) == est.D

def test_grad_wrong_before_fit():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        x = np.random.randn(est.D)
        
        for est in estimators:
            result = est.grad(x)
            assert_allclose(result, np.zeros(est.D))

def test_grad_wrong_input_type():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.fit(X)
        assert_raises(TypeError, est.grad, None)

def test_grad_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad, x)

def test_grad_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        x = np.random.randn(est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.grad, x)

def test_objective_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        est.objective(X)

def test_objective_result():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        result = est.objective(X)
        
        assert type(result) is np.float64

def test_objective_wrong_input_type():
    N = 10
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        assert_raises(TypeError, est.objective, None)

def test_objective_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.objective, Y)

def test_objective_wrong_input_dim():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        Y = np.random.randn(N, est.D + 1)
        
        est.fit(X)
        assert_raises(ValueError, est.objective, Y)

def test_xvalidate_objective_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        est.xvalidate_objective(X, num_folds=3, num_repetitions=1)

def test_xvalidate_objective_result():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        
        result = est.xvalidate_objective(X, num_folds=3, num_repetitions=2)
        
        assert type(result) is np.ndarray
        assert result.ndim == 2
        assert result.shape[0] == 2
        assert result.shape[1] == 3

def test_xvalidate_objective_wrong_input_type():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        assert_raises(TypeError, est.xvalidate_objective, X=None, num_folds=3, num_repetitions=2)
        assert_raises(TypeError, est.xvalidate_objective, X=X, num_folds=None, num_repetitions=2)
        assert_raises(TypeError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=None)
        
def test_xvalidate_objective_wrong_input_dim_X():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D, 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=2)

def test_xvalidate_objective_wrong_input_shape_X():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=2)

def test_xvalidate_objective_wrong_input_negative_int():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D + 1)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=0, num_repetitions=2)
        assert_raises(ValueError, est.xvalidate_objective, X=X, num_folds=3, num_repetitions=0)

def test_get_parameters_finite():
    N = 10
    names = get_instace_KernelExpFiniteGaussian(N).get_parameter_names()
    assert "sigma" in names
    assert "lmbda" in names
    assert len(names) == 2

def test_get_parameters_lite():
    N = 10
    names = get_instace_KernelExpLiteGaussian(N).get_parameter_names()
    assert "sigma" in names
    assert "lmbda" in names
    assert len(names) == 2

def test_get_parameters():
    N = 10
    estimators = get_estimator_instances(N)
    
    for estimator in estimators:
        param_dict = estimator.get_parameters()
        for name, value in param_dict.items():
            assert getattr(estimator, name) == value

def test_set_parameters_from_dict():
    N = 10
    estimators = get_estimator_instances(N)
    
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
    N = 10
    estimators = get_estimator_instances(N)
    
    for estimator in estimators:
        assert_raises(TypeError, estimator.set_parameters_from_dict, None)
        assert_raises(TypeError, estimator.set_parameters_from_dict, 1)
        assert_raises(TypeError, estimator.set_parameters_from_dict, [])
        
def test_set_parameters_from_dict_wrong_input_parameters():
    N = 10
    estimators = get_estimator_instances(N)
    
    for estimator in estimators:
        param_dict = estimator.get_parameters()
        param_dict['strange_parameter'] = 0
        assert_raises(ValueError, estimator.set_parameters_from_dict, param_dict)

def test_update_fit_if_exists_execute():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        if hasattr(est, 'update_fit'):
            X = np.random.randn(N, est.D)
            X2 = np.random.randn(N, est.D)
            est.fit(X)
        
            est.update_fit(X2)

def test_update_fit_if_exists_increasing_n():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        if hasattr(est, 'update_fit'):
            X = np.random.randn(N, est.D)
            X2 = np.random.randn(N, est.D)
            est.fit(X)
            old_n = est.n
            est.update_fit(X2)
            
            assert est.n == old_n + N

def test_update_fit_if_exists_equals_batch():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        if hasattr(est, 'update_fit'):
            x_test = np.random.randn(est.D)
            X = np.random.randn(N, est.D)
            X2 = np.random.randn(N, est.D)
            stacked = np.vstack((X, X2))
            est.fit(stacked)
            log_pdf_batch = est.log_pdf(x_test)
            grad_batch = est.grad(x_test)
            
            est.fit(X)
            est.update_fit(X2)
            log_pdf_online = est.log_pdf(x_test)
            grad_online = est.grad(x_test)
            
            assert_allclose(log_pdf_online, log_pdf_batch, err_msg=est.get_name())
            assert_allclose(grad_online, grad_batch, err_msg=est.__class__.__name__)

def test_update_fit_if_exists_wrong_input_type():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        if hasattr(est, 'update_fit'):
            assert_raises(TypeError, est.update_fit, None)
            assert_raises(TypeError, est.update_fit, 1)
            assert_raises(TypeError, est.update_fit, [1, 2, 3])
            
def test_update_fit_if_exists_wrong_input_shape():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        if hasattr(est, 'update_fit'):
            assert_raises(ValueError, est.update_fit, np.random.randn(N))
            assert_raises(ValueError, est.update_fit, np.random.randn(N, est.D - 1, 1))
            
def test_update_fit_if_exists_wrong_input_dims():
    N = 100
    estimators = get_estimator_instances(N)
    
    for est in estimators:
        X = np.random.randn(N, est.D)
        est.fit(X)
        
        if hasattr(est, 'update_fit'):
            assert_raises(ValueError, est.update_fit, np.random.randn(N, est.D + 1))
            assert_raises(ValueError, est.update_fit, np.random.randn(N, est.D - 1))
