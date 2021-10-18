from tensorflow.python import keras
# import onnxruntime
import pytest
import numpy as np
from hyperopt import hp
from scipy import stats

from mlsquare.models.embibe import rasch, fourPL
from sklearn.linear_model import LogisticRegression, LinearRegression

from mlsquare import registry
from test_architectures import _load_classification_data, _load_regression_data, _load_1PL_IrtData#_load_irt_data

def test_irt_Keras_regressor_basic_functionality():
    primal_model = rasch()
    proxy_model, mock_adapt = registry[('mlsquare', 'rasch')]['default']
    model = mock_adapt(proxy_model, primal_model)
    assert hasattr(model, 'fit') == True
    assert hasattr(model, 'coefficients') == True
    assert hasattr(model, 'predict') == True
    assert hasattr(model, 'save') == True

def test_irt_Keras_regressor_with_illformat_nas_params():
    primal_model = rasch()
    proxy_model, mock_adapt = registry[('mlsquare', 'rasch')]['default']
    model = mock_adapt(proxy_model, primal_model)

    di_error_config= {'ss_config1': (ValueError,[{
    "diff_params.kernel_params.mean": hp.uniform("diff_params.kernel_params.mean", -0.5,0.5)}]),
          'ss_config2': (TypeError, {
    "diff_params.kernel_params.mean": hp.uniform("diff_params.kernel_params.mean", [-0.5,0.5])}),
          'ss_config3': [BaseException, {
    "diff_params.kernel_params": hp.uniform("diff_params.kernel_params", -0.5,0.5)}],
         'ss_config4': [IndexError, {}]
          }
    di_error_msg= {ValueError: "Ill-format Search Space is passed/accepeted in nas_params",
    TypeError:"Boundary values in wrong format are passed/accepted in nas_params",
    BaseException: "Search keys given in wrong format are passed/accepted in nas_params",
    IndexError:"An empty Search Space is passed/accepted in nas_params"
    }
    x_train_u, x_train_q, y_train, _, _, _, _= _load_irt_data()
    for k, v in di_error_config.items():
        with pytest.raises(v[0]) as di_error_msg[v[0]]:
            nas_params= {'search_algo_name':'hyperOpt', 'search_space':v[1], 'union':False}
            model.fit(x_user= x_train_u, x_questions= x_train_q, y_vals= y_train, batch_size= 64, epochs=1, nas_params= nas_params)

@pytest.mark.xfail
def test_irt_keras_nas_regularizers():
    primal_model = fourPL()
    proxy_model, mock_adapt = registry[('mlsquare', 'fourPL')]['default']
    model = mock_adapt(proxy_model, primal_model)
    layer_nas_keys = dict(zip(['difficulty_level', 'disc_param', 'guessing_param',
    'slip_param'], ["diff_params.group_lasso.l1", "disc_params.group_lasso.l1",
    "guess_params.group_lasso.l1", "slip_params.group_lasso.l1"]))

    layer_names= np.random.choice(list(layer_nas_keys.keys()),2, replace=False)
    src_space= {
    layer_nas_keys[vals]: hp.uniform(layer_nas_keys[vals], 0, 10) for vals in layer_names}
    nas_params= {'search_algo_name':'hyperOpt', 'search_space':src_space, 'union':False}
    xtrain, y_train, x_train_user, x_train_questions=_load_1PL_IrtData(size=0.9)
    model.fit(x_user= x_train_user, x_questions= x_train_questions, y_vals= y_train, batch_size= 128, epochs=5, nas_params=nas_params)
    for lay in model.model.layers:
        if lay.name in layer_names:
            coefs= model.coefficients()[lay.name]
            true_params = np.zeros(coefs.shape) if lay.name!= 'disc_param' else np.ones(coefs.shape)
            _, pval= stats.ttest_rel(coefs, true_params)
            assert lay.activity_regularizer.l1!=0, "NAS didn't work, Activity regularizer is not updated"
            assert pval < 0.2, "group lasso did NOT work as expected, Try again adjusting l1/l2 bounds in NAS search space."


@pytest.mark.xfail
def test_irtkeras_regressor_with_nas_params():
    primal_model = rasch()
    proxy_model, mock_adapt = registry[('mlsquare', 'rasch')]['default']
    model = mock_adapt(proxy_model, primal_model)
    src_space= {
    "diff_params.kernel_params.mean": hp.uniform("diff_params.kernel_params.mean", 0.1,0.3)}
    nas_params= {'search_algo_name':'hyperOpt', 'search_space':src_space, 'union':False}
    x_train_u, x_train_q, y_train, _, _, _, _= _load_irt_data()
    model.fit(x_user= x_train_u, x_questions= x_train_q, y_vals= y_train, batch_size= 64, epochs=3, nas_params= nas_params)
    di = {lay.name:i for i, lay in enumerate(model.model.layers)}
    diff_mean = model.model.layers[di['difficulty_level']].kernel_initializer.mean
    assert diff_mean!=0, "Mean value isn't optimized, NAS did NOT traverse externel nas_params"
    for lay in ['difficulty_level', 'latent_trait/ability']:
        lay_stddev = model.model.layers[di[lay]].kernel_initializer.stddev
        assert lay_stddev==1, "Stddev value is also optimized, NAS(union=False) did NOT OVERRIDE default model_nas_params"
    assert isinstance(model.model, keras.engine.training.Model)

@pytest.mark.xfail
def test_irtkeras_regressor_with_nas_params_union():
    primal_model = rasch()
    proxy_model, mock_adapt = registry[('mlsquare', 'rasch')]['default']
    model = mock_adapt(proxy_model, primal_model)
    src_space= {
    "diff_params.kernel_params.mean": hp.uniform("diff_params.kernel_params.mean", 0.1,0.3)}
    nas_params= {'search_algo_name':'hyperOpt', 'search_space':src_space}
    x_train_u, x_train_q, y_train, _, _, _, _= _load_irt_data()
    model.fit(x_user= x_train_u, x_questions= x_train_q, y_vals= y_train, batch_size= 64, epochs=3, nas_params= nas_params)
    di = {lay.name:i for i, lay in enumerate(model.model.layers)}
    diff_mean = model.model.layers[di['difficulty_level']].kernel_initializer.mean
    assert diff_mean!=0, "Mean value isn't optimized, NAS did NOT traverse externel nas_params"
    for lay in ['difficulty_level', 'latent_trait/ability']:
        lay_stddev = model.model.layers[di[lay]].kernel_initializer.stddev
        assert lay_stddev!=1, "Stddev value isn't optimized, NAS did NOT traverse default model_nas_params"
    assert isinstance(model.model, keras.engine.training.Model)

def _run_adapter(dataset_loader, proxy_model, mock_adapt, primal_model):
    x_train, x_test, y_train, y_test = dataset_loader()
    model = mock_adapt(proxy_model, primal_model)
    params = {'optimizer':{'grid_search':['adam', 'nadam']}}
    epochs = 300
    batch_size = 50
    trained_model = model.fit(x_train, y_train, params=params, epochs=epochs, batch_size=batch_size)
    score = model.score(x_test, y_test)
    _pred = model.predict(x_test)
    assert isinstance(trained_model, keras.engine.sequential.Sequential)
    assert 0 <= score[1] <=1

def test_sklearn_keras_classifier_basic_functionality():
    primal_model = LogisticRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']
    model = mock_adapt(proxy_model, primal_model)
    assert hasattr(model, 'fit') == True
    assert hasattr(model, 'score') == True
    assert hasattr(model, 'save') == True

@pytest.mark.xfail() # Cross check why this is failing in circleCI.
def test_sklearn_keras_classifier_test_methods_with_params():
    primal_model = LogisticRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LogisticRegression')]['default']
    _run_adapter(_load_classification_data, proxy_model, mock_adapt, primal_model)

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

@pytest.mark.xfail()
def test_sklearn_keras_regressor_test_methods_with_params():
    primal_model = LinearRegression()
    proxy_model, mock_adapt = registry[('sklearn', 'LinearRegression')]['default']
    _run_adapter(_load_regression_data, proxy_model, mock_adapt, primal_model)

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
