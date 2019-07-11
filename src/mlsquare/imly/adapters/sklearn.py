#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from ..optmizers import get_best_model
import pickle
import onnxmltools
from numpy import where
from sklearn.preprocessing import OneHotEncoder

class SklearnKerasClassifier(KerasClassifier):
    def __init__(self, abstract_model, primal, **kwargs): ## Why use kwargs?
        # super(KerasClassifier, self).__init__(build_fn=build_fn)
        self.primal = primal
        self.params = None ## Temporary!
        self.abstract_model = abstract_model

    def fit(self, X, y, **kwargs): # Why kwargs?
        import numpy as np
        # Move to checkers function in commons.
        # if len(y.shape) == 1:
        #     units = 1
        # else:
        #     # units = y.shape[1]
        #     units = 2
        kwargs.setdefault('cuts_per_feature', None) ## Better way to handle?
        self.abstract_model.X = X
        self.abstract_model.y = y
        self.abstract_model.primal = self.primal

        self.abstract_model.cuts_per_feature = kwargs['cuts_per_feature']
        # self.abstract_model.update_params({'input_dim': X.shape[1]}) For dt
        # super(KerasClassifier, self).__init__(build_fn=self.abstract_model.create_model()) 
        # ## This is pointless!! build_fn is different from imly model created in tune
        # ## Change this.
        kwargs.setdefault('verbose', 0)
        verbose = kwargs['verbose']
        kwargs.setdefault('params', self.params)
        # Update as kwargs.setdefault('params', {})
        kwargs.setdefault('space', False)

        y = np.array(y)  # Compatibility with all formats?
        if len(y.shape) == 2 and y.shape[1] > 1:
            self.classes_ = np.arange(y.shape[1])
        elif (len(y.shape) == 2 and y.shape[1] == 1) or len(y.shape) == 1:
            self.classes_ = np.unique(y)
            y = np.searchsorted(self.classes_, y)
        else:
            raise ValueError(
                'Invalid shape for y: ' + str(y.shape))

        primal_model = self.primal
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)

        X, y, y_pred = self.abstract_model.transform_data(X, y, y_pred)
        ## Pushing this to wrapper brings back the old structure
        ## We will have to map classification models.

        primal_data = { ## Consider renaming -- primal_model_data or primal_results
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        ## Search for best model using Tune ##
        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model, primal_data=primal_data)
        super(KerasClassifier, self).__init__(build_fn=self.abstract_model.create_model())
        ## Incorrect implementation. Update this with the callable final model.
        return self.final_model  # Not necessary.

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.model, open(filename + '.pkl', 'wb'))

        self.model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

    ## Temporary hack. This should be handled during architectural refactoring.
    def score(self, x, y, **kwargs):
        if self.abstract_model.enc is not None:
            y = self.abstract_model.enc.transform(y)

        score = self.final_model.evaluate(x, y, **kwargs)
        return score

    def explain(self, **kwargs):
        # @param: SHAP or interpret
        print('Coming soon...')
        pass



class SklearnKerasRegressor(KerasRegressor):
    def __init__(self, abstract_model, primal, **kwargs):
        self.primal = primal
        self.abstract_model = abstract_model

    def fit(self, X, y, **kwargs):
        self.abstract_model.update_params({'input_dim': X.shape[1]})
        super(KerasRegressor, self).__init__(build_fn=self.abstract_model.create_model())
        # Check whether to compute for the best model or not
        kwargs.setdefault('verbose', 0)
        verbose = kwargs['verbose']
        # if (kwargs['params'] != self.params):
            # Optimize
        primal_model = self.primal
        primal_model.fit(X, y)
        y_pred = primal_model.predict(X)
        primal_data = {
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }

        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model, primal_data=primal_data)
        return self.final_model  # Not necessary.

    def score(self, x, y, **kwargs):
        score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
        # keras_regressor treats all score values as loss and adds a '-ve' before passing
        return -score

    def save(self, filename=None):
        if filename == None:
            raise ValueError(
                'Name Error: to save the model you need to specify the filename')

        pickle.dump(self.model, open(filename + '.pkl', 'wb'))

        self.model.save(filename + '.h5')

        onnx_model = onnxmltools.convert_keras(self.model)
        onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

'''
Updates -
1) Remove parent Keras wrapper - IMP
'''