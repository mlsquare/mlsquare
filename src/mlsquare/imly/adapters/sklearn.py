#!/usr/bin/env python
# -*- coding: utf-8 -*-
from ..optmizers import get_best_model
from ..utils.functions import _parse_params 
from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
import pickle
import onnxmltools
from numpy import where
from sklearn.preprocessing import OneHotEncoder
import numpy as np
## Python log module - log_info, log_error, log_debug, log_warn -- Data type conversion scenario

class SklearnKerasClassifier():
    def __init__(self, abstract_model, primal, **kwargs):
        self.primal = primal
        self.params = None ## Temporary!
        self.abstract_model = abstract_model

    def fit(self, X, y, **kwargs):
        kwargs.setdefault('cuts_per_feature', None) ## Better way to handle?

        self.abstract_model.cuts_per_feature = kwargs['cuts_per_feature'] ## For all models?
        kwargs.setdefault('verbose', 0)
        verbose = kwargs['verbose']
        kwargs.setdefault('params', self.params)
        kwargs.setdefault('space', False)
        self.params = kwargs['params']
        X = np.array(X)
        y = np.array(y)

        primal_model = self.primal
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)

        X, y, y_pred = self.abstract_model.transform_data(X, y, y_pred)

        # This should happen only after transformation.
        self.abstract_model.X = X
        self.abstract_model.y = y
        self.abstract_model.primal = self.primal

        if self.params != None: ## Validate implementation with different types of tune input
            self.params = _parse_params(self.params, return_as='flat')
            self.abstract_model.update_params(self.params)

        primal_data = { ## Consider renaming -- primal_model_data or primal_results
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        ## Search for best model using Tune ##
        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model, primal_data=primal_data)
        return self.final_model  # Not necessary.

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.final_model, open(filename + '.pkl', 'wb'))

        self.final_model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.final_model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

    def score(self, X, y, **kwargs):
        if self.abstract_model.enc is not None:
            ## Should we accept pandas?
            y = np.array(y)
            X = np.array(X)
            if len(y.shape) == 1 or y.shape[1] == 1:
                y = self.abstract_model.enc.transform(y.reshape(-1,1))
                y = y.toarray() ## Cross check with logistic regression flow
            else:
                y = self.abstract_model.enc.transform(y)
                y = y.toarray()
        score = self.final_model.evaluate(X, y, **kwargs)
        return score

    def explain(self, **kwargs):
        # @param: SHAP or interpret
        print('Coming soon...')
        pass



class SklearnKerasRegressor():
    def __init__(self, abstract_model, primal, **kwargs):
        self.primal = primal
        self.abstract_model = abstract_model
        self.params = None

    def fit(self, X, y, **kwargs):
        self.abstract_model.X = X
        self.abstract_model.y = y
        self.abstract_model.primal = self.primal
        kwargs.setdefault('verbose', 0)
        kwargs.setdefault('params', self.params)
        verbose = kwargs['verbose']
        self.params = kwargs['params']
        
        if self.params != None: ## Validate implementation with different types of tune input
            self.params = _parse_params(self.params, return_as='flat')
            self.abstract_model.update_params(self.params)
        primal_model = self.primal
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)
        primal_data = {
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model, primal_data=primal_data)
        return self.final_model  # Not necessary.

    def score(self, X, y, **kwargs):
        score = self.final_model.evaluate(X, y, **kwargs)
        return score

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.final_model, open(filename + '.pkl', 'wb'))

        self.final_model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.final_model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')
