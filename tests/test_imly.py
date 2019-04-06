#!/usr/bin/env python
# -*- coding: utf-8 -*-

__author__ = "Ravi Mula"
__copyright__ = "MLSquare"
__license__ = "MIT"


import pytest, os, json

import pandas as pd
import numpy as np
import copy

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix
from sklearn.metrics import mean_squared_error

from mlsquare.imly import dope
from datasets import _load_diabetes, _load_airfoil, _load_iris, _load_salary


def _imly(model, dataset, test_size = 0.60, using='dnn', best=True, **kwargs):
    X, Y = dataset()
    x_train, x_test, y_train, y_test = train_test_split(
        X, Y, test_size=0.60, random_state=0)

    primal_model = copy.deepcopy(model)

    # Fit Primal Model
    primal_model.fit(x_train, y_train.values.ravel()) # Why use '.values.ravel()'? --
    y_pred = primal_model.predict(x_test)

    if (primal_model.__class__.__name__ == 'LogisticRegression') or \
       (primal_model.__class__.__name__ == 'LinearDiscriminantAnalysis'):
        primal_score = primal_model.score(x_test, y_test)
    else:
        primal_score = mean_squared_error(y_test, y_pred)

    # Fit DNN model
    m = dope(model, using=using, best=best)
    m.fit(x_train, y_train.values.ravel())
    keras_score = m.score(x_test, y_test)
    if (using == 'dnn'):
        m.save('test_save_method')
    print ("========= OUTPUT OF RESULT =========\n")
    print ("Args Passed: \n")
    print ("Model: %s\n" % model.__class__.__name__)
    print ("Using: %s\n" % using)
    print ("Best: %s\n" % best)
    print ("Primal Score: %s\n" % (primal_score))
    print ("Keras Score: %s \n " % (keras_score))
    print ("========= END OF RESULT =========\n")
    return primal_score, keras_score

def _run_multiple(model, datasets):
    for dataset in datasets:
        # No DNN
        primal_score_0, keras_score_0 = _imly(
            model=model, dataset=dataset, using=None, best=False)

        # DNN, No Best
        primal_score_1, keras_score_1 = _imly(
            model=model, dataset=dataset, using='dnn', best=False)

        # DNN, Best
        primal_score_2, keras_score_2 = _imly(
            model=model, dataset=dataset, using='dnn', best=True)

        assert 0 <= primal_score_0 <= 1
        assert 0 <= keras_score_0 <= 1
        assert 0 <= primal_score_1 <= 1
        assert 0 <= keras_score_1 <= 1
        assert 0 <= primal_score_2 <= 1
        assert 0 <= keras_score_2 <= 1

def test_linear_regression():
    datasets = [_load_diabetes]
    model = LinearRegression()
    _run_multiple (model, datasets)

def test_logistic_regression():
    datasets = [_load_iris, _load_salary]
    model = LogisticRegression()
    _run_multiple(model, datasets)
