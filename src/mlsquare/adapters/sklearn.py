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
        kwargs.setdefault('epochs', 250)
        kwargs.setdefault('batch_size', 30)
        self.params = kwargs['params']
        X = np.array(X)
        y = np.array(y)

        primal_model = self.primal
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)

        X, y, y_pred = self.abstract_model.transform_data(X, y, y_pred)

        # This should happen only after transformation.
        self.abstract_model.X = X ##  abstract -> model_skeleton
        self.abstract_model.y = y
        self.abstract_model.primal = self.primal

        if self.params != None: ## Validate implementation with different types of tune input
            if type(self.params) != dict:
                raise TypeError("Params should be of type 'dict'")
            self.params = _parse_params(self.params, return_as='flat')
            self.abstract_model.update_params(self.params)

        primal_data = { ## Consider renaming -- primal_model_data or primal_results
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        ## Search for best model using Tune ##
        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model,
                                            primal_data=primal_data, epochs=kwargs['epochs'], batch_size=kwargs['batch_size'])
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

    def predict(self, X):
        X = np.array(X)
        if hasattr(self.final_model, 'predict_classes'):
            pred = self.final_model.predict_classes(X)
        else:
            pred = self.final_model.predict(X)
            pred = np.argmax(pred,axis=1)
        return pred

    def predict_proba(self, X):
        pass

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

    def predict(self, X):
        '''
        Pending:
        1) Write a 'filter_sk_params' function(check keras_regressor wrapper) if necessary.
        2) Data checks and data conversions
        '''
        pred = self.final_model.predict(X)
        return pred

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.final_model, open(filename + '.pkl', 'wb'))

        self.final_model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.final_model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

class SklearnPytorchClassifier():
    def __init__(self, abstract_model, primal, **kwargs):
        self.primal = primal
        self.params = None ## Temporary!
        self.abstract_model = abstract_model

    def fit(self, X, y, **kwargs):
        self.abstract_model.X = X
        self.abstract_model.y = y
        self.abstract_model.primal = self.primal

        for epoch in range(50):
            # Forward Propagation
            # Access model, criterion and optimizer from abstract_model
            # Alter how tune computes `fit`. Override keras_model.fit option
            y_pred = model(x)    # Compute and print loss
            loss = criterion(y_pred, y)
            print('epoch: ', epoch,' loss: ', loss.item())    # Zero the gradients
            optimizer.zero_grad()

            # perform a backward pass (backpropagation)
            loss.backward()

            # Update the parameters
            optimizer.step()

'''
1) test data_generators -- keras article
    + sklearn's capacity to deal with generators
    + Adding data_generators support to the proxy_model
    + Both pytorch and keras
2) Idempotency
    + Saving followed by load - run and validate the results
    + Choose 'using' = None. Returns the primal model
    + A fallback option - Status code, primal and info
'''