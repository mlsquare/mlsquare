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
from datasets import _load_diabetes, _load_iris



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

@pytest.mark.xfail()
def test_linear_regression_ttest():
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = LinearRegression()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)

    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_logistic_regression_chisquare():
    ## Set np seed
    x_train, x_test, y_train, _ = _load_classification_data()
    primal_model = LogisticRegression()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred2,pred1)

    assert p_value > 0.8

@pytest.mark.xfail()
def test_ridge_ttest():
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = Ridge()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)

    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_lasso_ttest():
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = Lasso()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)

    assert p_value[0] < 0.1


@pytest.mark.xfail()
def test_elasticnet_ttest():
    x_train, x_test, y_train, _ = _load_regression_data()
    primal_model = ElasticNet()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)

    _, p_value = stats.ttest_rel(pred2,pred1)

    assert p_value[0] < 0.1

@pytest.mark.xfail()
def test_linear_svc_chisquare():
    ## Set np seed
    x_train, x_test, y_train, _ = _load_classification_data()
    primal_model = LinearSVC()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred2,pred1)

    assert p_value > 0.8

@pytest.mark.xfail()
def test_svc_chisquare():
    ## Set np seed
    x_train, x_test, y_train, _ = _load_classification_data()
    primal_model = SVC()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred2,pred1)

    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_using_chisquare():
    ## Set np seed
    x_train, x_test, y_train, _ = _load_classification_data()
    y_train = to_categorical(y_train)
    primal_model = DecisionTreeClassifier()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    pred2 = np.argmax(pred2, axis=1)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred1,pred2)

    assert p_value > 0.8

@pytest.mark.xfail()
def test_decision_trees_with_cuts_using_chisquare():
    ## Set np seed
    x_train, x_test, y_train, _ = _load_classification_data()
    y_train = to_categorical(y_train)
    primal_model = DecisionTreeClassifier()
    proxy_model = _mock_dope(primal_model)
    proxy_model.fit(x_train,y_train,cuts_per_feature=2, epochs=300)

    primal_model.fit(x_train, y_train)
    pred1, pred2 = _predict_primal_and_proxy(proxy_model, primal_model, x_test)
    pred2 = np.argmax(pred2, axis=1)
    pred1 = pred1.reshape(pred2.shape)
    pred2 += 1
    pred1 += 1
    _, p_value = stats.chisquare(pred1,pred2)

    assert p_value > 0.8

def test_glm_create_model_method():
    class MockModel(GeneralizedLinearModel):
        def __init__(self):
            self.adapter = SklearnKerasClassifier
            self.module_name = 'sklearn'
            self.name = 'LogisticRegression'
            self.version = 'default'
            model_params = {'layer_1': {'l1': 0,
                                        'l2': 0,
                                        'activation': 'sigmoid'},
                            'optimizer': 'adam',
                            'loss': 'binary_crossentropy'
                            }

            self.set_params(params=model_params, set_by='model_init')

    mock_model = MockModel()
    mock_model.y = np.array([[0,1], [0,1]])
    mock_model.X = np.array([[0,1], [0,1]])
    mock_proxy_model = mock_model.create_model()

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 2
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'binary_crossentropy'


def test_kernel_glm_create_model_method():
    class MockModel(KernelGeneralizedLinearModel):
        def __init__(self):
            self.adapter = SklearnKerasClassifier
            self.module_name = 'sklearn'
            self.name = 'SVC'
            self.version = 'default'
            model_params = {'layer_1': {'kernel_dim': 10,  # Make it 'units' -- Why?
                                        'activation': 'linear'
                                        },
                            'layer_2': {
                                        'activation': 'softmax'
                                        },
                            'optimizer': 'adam',
                            'loss': 'categorical_hinge'}

            self.set_params(params=model_params, set_by='model_init')

    mock_model = MockModel()
    mock_model.y = np.array([[0,1], [0,1]])
    mock_model.X = np.array([[0,1], [0,1]])
    mock_proxy_model = mock_model.create_model()

    assert mock_proxy_model.get_config()['layers'][0]['config']['activation'] == 'linear'
    assert mock_proxy_model.get_config()['layers'][0]['config']['units'] == 10
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_hinge'

def test_cart_create_model_method():
    class MockModel(CART):
        def __init__(self):
            self.cuts_per_feature = None
            self.adapter = SklearnKerasClassifier
            self.module_name = 'sklearn'
            self.name = 'DecisionTreeClassifier'
            self.version = 'default'
            model_params = {
                'layer_3': {'activation': 'sigmoid'},
                'optimizer': 'adam',
                'loss': 'categorical_crossentropy'
            }

            self.set_params(params=model_params, set_by='model_init')

    mock_model = MockModel()
    mock_model.y = np.array([[0,1], [0,1]])
    mock_model.X = np.array([[0,1], [0,1]])
    x_train, _, y_train, _ = _load_classification_data()
    y_train = to_categorical(y_train)
    mock_model.primal = DecisionTreeClassifier().fit(x_train, y_train)
    mock_proxy_model = mock_model.create_model()

    assert mock_proxy_model.get_config()['layers'][2]['config']['activation'] == 'sigmoid'
    assert mock_proxy_model.optimizer.__class__.__name__ == 'Adam'
    assert mock_proxy_model.loss == 'categorical_crossentropy'

def test_linear_svc_transform_data():
    # Pending
    pass
