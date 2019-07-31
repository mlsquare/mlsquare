import keras
# import onnxruntime
import pytest
import numpy as np
from scipy import stats

from sklearn.linear_model import LogisticRegression, LinearRegression

from mlsquare import registry
from test_architectures import _load_classification_data, _load_regression_data

def test_sklearn_keras_classifier_basic_functionality():
    primal_model = LogisticRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']
    model = mock_adapt(proxy_model, primal_model)
    assert hasattr(model, 'fit') == True
    assert hasattr(model, 'score') == True
    assert hasattr(model, 'save') == True

def test_sklearn_keras_classifier_test_methods_with_params():
    primal_model = LogisticRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']

    x_train, x_test, y_train, y_test = _load_classification_data()
    model = mock_adapt(proxy_model, primal_model)
    params = {'optimizer':{'grid_search':['adam', 'nadam']}}
    epochs = 300
    batch_size = 50
    trained_model = model.fit(x_train, y_train, params=params, epochs=epochs, batch_size=batch_size)
    score = model.score(x_test, y_test)
    _pred = model.predict(x_test)
    assert isinstance(trained_model, keras.engine.sequential.Sequential)
    assert 0 <= score[1] <=1

# def test_sklearn_keras_classifier_test_save():
#     primal_model = LogisticRegression()
#     proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']

#     x_train, x_test, y_train, _ = _load_classification_data()
#     model = mock_adapt(proxy_model, primal_model)
#     epochs = 300
#     batch_size = 50
#     _trained_model = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
#     model.save('test_onnx')
#     # onnx_model = onnx.load("test_onnx.onnx")
#     sess = onnxruntime.InferenceSession('test_onnx.onnx')
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     x_test = x_test.values.astype(np.float32)
#     result_as_proba = sess.run([output_name], {input_name: x_test})
#     result_as_classes = (result_as_proba[0]>0.6).astype(np.int)
#     keras_model_pred = model.predict(x_test)
#     result_as_classes += 1
#     keras_model_pred += 1
#     _, p_value = stats.chisquare(result_as_classes, keras_model_pred)
#     assert p_value > 0.9

def test_sklearn_keras_classifier_with_inappropriate_params():
    primal_model = LogisticRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']

    x_train, _, y_train, _ = _load_classification_data()
    model = mock_adapt(proxy_model, primal_model)
    params = [{'optimizer':{'grid_search':['adam', 'nadam']}}]
    with pytest.raises(TypeError) as _:
        _trained_model = model.fit(x_train, y_train, params=params)

def test_sklearn_keras_regressor_basic_functionality():
    primal_model = LinearRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LinearRegression')]['default']
    model = mock_adapt(proxy_model, primal_model)
    assert hasattr(model, 'fit') == True
    assert hasattr(model, 'score') == True
    assert hasattr(model, 'save') == True

def test_sklearn_keras_regressor_test_methods_with_params():
    primal_model = LinearRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LinearRegression')]['default']

    x_train, x_test, y_train, y_test = _load_regression_data()
    model = mock_adapt(proxy_model, primal_model)
    params = {'optimizer':{'grid_search':['adam', 'nadam']}}
    epochs = 300
    batch_size = 50
    trained_model = model.fit(x_train, y_train, params=params, epochs=epochs, batch_size=batch_size)
    score = model.score(x_test, y_test)
    _pred = model.predict(x_test)
    assert isinstance(trained_model, keras.engine.sequential.Sequential)
    assert 0 <= score[1] <=1

def test_sklearn_keras_regressor_with_inappropriate_params():
    primal_model = LinearRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LinearRegression')]['default']

    x_train, _, y_train, _ = _load_regression_data()
    model = mock_adapt(proxy_model, primal_model)
    params = [{'optimizer':{'grid_search':['adam', 'nadam']}}]
    with pytest.raises(TypeError) as _:
        _trained_model = model.fit(x_train, y_train, params=params)

# @pytest.mark.xfail()
# def test_sklearn_keras_regressor_test_save():
#     # Rewrite this test. This should not be non-deterministic.
#     primal_model = LinearRegression()
#     proxy_model, mock_adapt = registry[('sklearn', 'LinearRegression')]['default']

#     x_train, x_test, y_train, _ = _load_regression_data()
#     model = mock_adapt(proxy_model, primal_model)
#     epochs = 300
#     batch_size = 50
#     _trained_model = model.fit(x_train, y_train, epochs=epochs, batch_size=batch_size)
#     model.save('test_onnx')
#     # onnx_model = onnx.load("test_onnx.onnx")
#     sess = onnxruntime.InferenceSession('test_onnx.onnx')
#     input_name = sess.get_inputs()[0].name
#     output_name = sess.get_outputs()[0].name
#     x_test = x_test.values.astype(np.float32)
#     result = sess.run([output_name], {input_name: x_test})
#     # result_as_classes = (result_as_proba[0]>0.6).astype(np.int)
#     keras_model_pred = model.predict(x_test)
#     _, p_value = stats.ttest_rel(result[0], keras_model_pred)
#     assert p_value[0] < 0.1
