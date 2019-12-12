import pytest

import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import LinearSVC, SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

from keras.utils import to_categorical

from mlsquare.base import registry
from mlsquare.adapters import SklearnKerasClassifier
from mlsquare.architectures.sklearn import GeneralizedLinearModel, KernelGeneralizedLinearModel, CART
from datasets import _load_diabetes, _load_iris, _load_boston
from sklearn.decomposition import TruncatedSVD
import tensorflow as tf

def _load_decomposition_data():
    X, Y = _load_boston()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train

def _load_regression_data():
    X, Y = _load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, x_test, y_train, y_test

def _load_classification_data():
    X, Y = _load_iris()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, x_test, y_train, y_test

def _mock_dope(primal_model):
    model_skeleton, adapt = registry[('sklearn', primal_model.__class__.__name__)]['default']
    final_model = adapt(model_skeleton, primal_model)
    return final_model

def _predict_primal_and_proxy(proxy, primal, x_test):
    pred1 = proxy.predict(x_test)
    pred2 = primal.predict(x_test)
    return pred1, pred2

def _run_ttest(primal_model_class, epochs):
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = primal_model_class()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=epochs)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)
    return p_value

def _run_chisquare(primal_model_class, epochs, cuts_per_feature=None):
    x_train, x_test, y_train, _ = _load_classification_data()
    primal_model = primal_model_class()
    if isinstance(primal_model, DecisionTreeClassifier):
        y_train = to_categorical(y_train)
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=epochs, cuts_per_feature=cuts_per_feature)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    if isinstance(primal_model, DecisionTreeClassifier):
        pred2 = np.argmax(pred2, axis=1)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred2,pred1)
    return p_value

def _prepare_mock_model(parent_class, adapter, module_name, model_name, version, model_params, primal_model=None):
    class MockModel(parent_class):
        def __init__(self):
            self.cuts_per_feature = None
            self.adapter = adapter
            self.module_name = module_name
            self.name = model_name
            self.version = version

            self.set_params(params=model_params, set_by='model_init')

    mock_model = MockModel()
    mock_model.y = np.array([[0,1], [0,1]])
    mock_model.X = np.array([[0,1], [0,1]])
    if isinstance(primal_model, DecisionTreeClassifier):
        x_train, _, y_train, _ = _load_classification_data()
        y_train = to_categorical(y_train)
        mock_model.primal = DecisionTreeClassifier().fit(x_train, y_train)
    mock_proxy_model = mock_model.create_model()
    return mock_proxy_model

def _run_decomposition_test(primal_model_class, num_components):
    X = _load_decomposition_data()
    primal_model = primal_model_class(n_components= num_components)
    proxy_model = _mock_dope(primal_model)
    sess= tf.Session()

    tf_trans_x = proxy_model.fit_transform(X)
    tf_sigma = proxy_model.singular_values_ 
    tf_U = tf_trans_x/tf_sigma
    tf_V= proxy_model.components_
    tf_approx_recon =  sess.run(tf.matmul(tf_U, tf.matmul(tf.linalg.diag(tf_sigma), tf_V)))#, adjoint_b=True) v.T in arch-line#161
    #tf_approx_recon= np.dot(tf_U, np.dot(np.diag(tf_sigma), tf_V))#Since tf_U/tf_V are not tensors anymore

    skl_trans_x = primal_model.fit_transform(X)
    skl_sigma = primal_model.singular_values_    
    skl_U = skl_trans_x/skl_sigma
    skl_V= primal_model.components_
    skl_approx_recon = np.dot(skl_U, np.dot(np.diag(skl_sigma), skl_V))

    result = np.allclose(tf_approx_recon, skl_approx_recon)
    _, p_value = stats.ttest_rel(tf_sigma, skl_sigma)

    return result, p_value

def test_svd_reconstruction():
    result, _ = _run_decomposition_test(TruncatedSVD, 10)
    assert result is True

def test_svd_sigma_vals():
    _, p_value = _run_decomposition_test(TruncatedSVD, 10)
    assert p_value < 1e-01

@pytest.mark.xfail()
def test_linear_regression_ttest():
    p_value = _run_ttest(LinearRegression, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_logistic_regression_chisquare():
    p_value = _run_chisquare(LogisticRegression, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_ridge_ttest():
    p_value = _run_ttest(Ridge, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_lasso_ttest():
    p_value = _run_ttest(Lasso, 300)
    assert p_value[0] < 0.1


@pytest.mark.xfail()
def test_elasticnet_ttest():
    p_value = _run_ttest(ElasticNet, 300)
    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_linear_svc_chisquare():
    p_value = _run_chisquare(LinearSVC, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_svc_chisquare():
    p_value = _run_chisquare(SVC, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_using_chisquare():
    p_value = _run_chisquare(DecisionTreeClassifier, 300)
    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_with_cuts_using_chisquare():
    p_value = _run_chisquare(DecisionTreeClassifier, 300, cuts_per_feature=2)
    assert p_value > 0.8

def test_glm_create_model_method():
    model_params = {'layer_1': {'l1': 0,
                                'l2': 0,
                                'activation': 'sigmoid'},
                    'optimizer': 'adam',
                    'loss': 'binary_crossentropy'
                    }
    mock_proxy_model = _prepare_mock_model(GeneralizedLinearModel, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params)

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 2
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'binary_crossentropy'


def test_kernel_glm_create_model_method():
    model_params = {'layer_1': {'kernel_dim': 10,
                                'activation': 'linear'
                                },
                    'layer_2': {
                                'activation': 'softmax'
                                },
                    'optimizer': 'adam',
                    'loss': 'categorical_hinge'}
    mock_proxy_model = _prepare_mock_model(KernelGeneralizedLinearModel, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params)

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'linear'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 10
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_hinge'

def test_cart_create_model_method():
    model_params = {
        'layer_3': {'activation': 'sigmoid'},
        'optimizer': 'adam',
        'loss': 'categorical_crossentropy'
    }

    mock_proxy_model = _prepare_mock_model(CART, SklearnKerasClassifier,
                                           'sklearn', 'LogisticRegression',
                                           'default', model_params, primal_model=DecisionTreeClassifier())

    assert mock_proxy_model.get_config()['layers'][2]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_crossentropy'

def test_linear_svc_transform_data():
    # Pending
    pass
