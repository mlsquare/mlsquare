#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Ravi Mula"
__copyright__ = "MLSquare"
__license__ = "MIT"


import pytest, os, json

import pandas as pd
import numpy as np
import copy

from numpy.testing import assert_array_almost_equal, assert_allclose, assert_approx_equal, assert_raises

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from mlsquare.imly.base import registry
from mlsquare.imly import dope
from mlsquare.imly.adapters import SklearnKerasClassifier, SklearnKerasRegressor
from datasets import _load_diabetes, _load_airfoil, _load_iris, _load_salary


# def _imly(model, dataset, test_size = 0.60, **kwargs):
#     X, Y = dataset()
#     x_train, x_test, y_train, y_test = train_test_split(
#         X, Y, test_size=0.60, random_state=0)

#     primal_model = copy.deepcopy(model)

#     # Fit Primal Model
#     primal_model.fit(x_train, y_train.values.ravel()) # Why use '.values.ravel()'? --
#     y_pred = primal_model.predict(x_test)

#     if (primal_model.__class__.__name__ == 'LogisticRegression') or \
#        (primal_model.__class__.__name__ == 'LinearDiscriminantAnalysis'):
#         primal_score = primal_model.score(x_test, y_test)
#     else:
#         primal_score = mean_squared_error(y_test, y_pred)

#     # Fit DNN model
#     m = dope(model, using=using, best=best)
#     m.fit(x_train, y_train.values.ravel())
#     keras_score = m.score(x_test, y_test)
#     if (using == 'dnn'):
#         m.save('test_save_method')
#     print ("========= OUTPUT OF RESULT =========\n")
#     print ("Args Passed: \n")
#     print ("Model: %s\n" % model.__class__.__name__)
#     print ("Using: %s\n" % using)
#     print ("Best: %s\n" % best)
#     print ("Primal Score: %s\n" % (primal_score))
#     print ("Keras Score: %s \n " % (keras_score))
#     print ("========= END OF RESULT =========\n")
#     return primal_score, keras_score

# def _run_multiple(model, datasets):
#     for dataset in datasets:

#         primal_score_0, keras_score_0 = _imly(model=model, dataset=dataset)

#         # # DNN, No Best
#         # primal_score_1, keras_score_1 = _imly(
#         #     model=model, dataset=dataset, using='dnn', best=False)

#         # # DNN, Best
#         # primal_score_2, keras_score_2 = _imly(
#         #     model=model, dataset=dataset, using='dnn', best=True)

#         assert 0 <= primal_score_0 <= 1
#         assert 0 <= keras_score_0 <= 1
#         assert 0 <= primal_score_1 <= 1
#         assert 0 <= keras_score_1 <= 1
#         assert 0 <= primal_score_2 <= 1
#         assert 0 <= keras_score_2 <= 1

def _load_classification_data():
    X, Y = _load_iris()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    return x_train, x_test, y_train, y_test

def _replicate_dope(primal_model):
    model_skeleton, adapt = registry[('sklearn', primal_model.__class__.__name__)]['default']
    final_model = adapt(model_skeleton, primal_model)
    return final_model

def _train_and_score_model(dataset, module_name, model_name):
    X, Y = dataset()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

    abstract_model, adapter = registry[('sklearn', 'LogisticRegression')]['default']
    abstract_model.y = y_train
    abstract_model.X = x_train
    final_model = abstract_model.create_model()

    final_model.fit(x_train, y_train)
    score = final_model.evaluate(x_test, y_test)[1]
    assert abstract_model.adapter == adapter
    assert abstract_model.module_name == 'sklearn' 
    assert abstract_model.name == 'LogisticRegression'
    assert abstract_model.version == 'default'
    assert 0 <= score <= 1

# def test_linear_regression():
#     datasets = [_load_diabetes]
#     model = LinearRegression()
#     _run_multiple (model, datasets)
@pytest.mark.skip(reason="Skipping...")
def test_logistic_regression():
    # datasets = [_load_iris, _load_salary]
    # datasets = _load_iris
    # model = LogisticRegression()
    # _run_multiple(model, datasets)
    _train_and_score_model(_load_iris, 'sklearn', 'LogisticRegression')
    # X, Y = _load_iris()
    # x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

    # abstract_model, _ = registry[('sklearn', 'LogisticRegression')]['default']
    # abstract_model.y = y_train
    # abstract_model.X = x_train
    # final_model = abstract_model.create_model()

    # final_model.fit(x_train, y_train)
    # score = final_model.evaluate(x_test, y_test)[1]
    # assert abstract_model.adapter == SklearnKerasClassifier
    # assert abstract_model.module_name == 'sklearn' 
    # assert abstract_model.name == 'LogisticRegression'
    # assert abstract_model.version == 'default'
    # assert 0 <= score <= 1

def test_linear_svc():
    # datasets = [_load_iris, _load_salary]
    # datasets = _load_iris
    # model = LogisticRegression()
    # _run_multiple(model, datasets)
    X, Y = _load_iris()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)

    abstract_model, _ = registry[('sklearn', 'LinearSVC')]['default']
    abstract_model.y = y_train
    abstract_model.X = x_train
    final_model = abstract_model.create_model()

    final_model.fit(x_train, y_train)
    score = final_model.evaluate(x_test, y_test)[1]
    assert abstract_model.adapter == SklearnKerasClassifier
    assert abstract_model.module_name == 'sklearn' 
    assert abstract_model.name == 'LinearSVC'
    assert abstract_model.version == 'default'
    assert 0 <= score <= 1

@pytest.mark.skip(reason="Skipping...")
def test_linear_regression():
    # datasets = [_load_iris, _load_salary]
    # datasets = _load_iris
    # model = LogisticRegression()
    # _run_multiple(model, datasets)
    X, Y = _load_diabetes()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    abstract_model, _ = registry[('sklearn', 'LinearRegression')]['default']

    abstract_model.y = y_train
    abstract_model.X = x_train

    final_model = abstract_model.create_model()

    final_model.fit(x_train, y_test)
    score = final_model.evaluate(x_test, y_test)
    assert abstract_model.adapter == SklearnKerasRegressor
    assert abstract_model.module_name == 'sklearn' 
    assert abstract_model.name == 'LinearRegression'
    assert abstract_model.version == 'default'
    assert 0 <= score <= 1
@pytest.mark.skip(reason="Skipping...")
def test_logistic_regression_primal_and_proxy_concordance():
    X, Y = _load_iris()
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.60, random_state=0)
    primal_model = LogisticRegression()
    model_skeleton, adapt = registry[('sklearn', 'LogisticRegression')]['default']
    final_model = adapt(model_skeleton, primal_model)
    final_model.fit(x_train,y_train, epochs=300)
    pred1 = final_model.score(x_test, y_test)[1]
    pred2 = primal_model.fit(x_train, y_train).score(x_test, y_test)
    assert_array_almost_equal(pred1, pred2, 1) # abs(desired-actual) < 1.5 * 10**(-decimal)
    assert_allclose(pred1, pred2, rtol=1e-1, atol=1) # abs(desired-actual) < atol + rtol * abs(desired)
    assert_approx_equal(pred1, pred2,1) # significant number
    ## test for normality of the residuals
    ## SPC - Variance checks
    ## Not determinstic test
    ## Paired t-test(Check scipy stat tests)
    ## Accuracy for classification algos
    ## Provide seed value
    ## Prioritize obvious errors from user

@pytest.mark.skip(reason="Skipping...")
def test_logistic_regression_with_external_parameters():
    ## Both wrong params and right ones
    x_train, x_test, y_train, y_test = _load_classification_data()
    primal_model = LogisticRegression()
    final_model = _replicate_dope(primal_model)
    params = {'activation': {'grid_search':['adam', 'nadam']}}
    final_model.fit(x_train,y_train, params=params, epochs=300)
    pred1 = final_model.score(x_test, y_test)[1]
    pred2 = primal_model.fit(x_train, y_train).score(x_test, y_test)
    assert_array_almost_equal(pred1, pred2, 1)


# @pytest.mark.skip(reason="Skipping...")
def test_logistic_regression_exception():
    ## Both wrong params and right ones
    x_train, x_test, y_train, y_test = _load_classification_data()
    primal_model = LogisticRegression()
    final_model = _replicate_dope(primal_model)
    params = {'optimizer': ['adam', 'nadam']}
    final_model.fit(x_train,y_train, params=params, epochs=300)
    assert_raises(TypeError, final_model.fit(), x_train,y_train, params=params, epochs=300)
    # pred1 = final_model.score(x_test, y_test)[1]
    # pred2 = primal_model.fit(x_train, y_train).score(x_test, y_test)
    # assert_array_almost_equal(pred1, pred2, 1) 

@pytest.mark.skip(reason="Skipping...")
def test_update_params():
    model_skeleton, _ = registry[('sklearn', 'LogisticRegression')]['default']
    default_params = model_skeleton.get_params()
    test_params = {'new_param1': 1, 'new_param2':2}
    model_skeleton.update_params(test_params)
    assert default_params.update(test_params) == model_skeleton.get_params()


# def test_get_params():
#     ## From BaseModel class
#     pass



def test_dope(): # Change description. More specific about the what is being tested
    model = LogisticRegression()
    m = dope(model)
    assert hasattr(m, 'fit') == True
    assert hasattr(m, 'score') == True
    assert hasattr(m, 'save') == True

def test_dope_external_model(): # test_importing_external_modules
    model = LogisticRegression()
    abstract_model, adapter = registry[('sklearn', 'LogisticRegression')]['default']
    m = dope(model, abstract_model=abstract_model, adapter=adapter)

    assert hasattr(m, 'fit') == True
    assert hasattr(m, 'score') == True
    assert hasattr(m, 'save') == True

'''
1) Cross check sklearn and numpy testing routines
2) How to practise TDD while developing MLSqaure package?
'''
