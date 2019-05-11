#!/usr/bin/env python
# -*- coding: utf-8 -*-

from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
from ..optmizers import get_best_model
import pickle
import onnxmltools

# Mixing - multi inheritence
# @abstractclass
class GetModelAbstract(**kwargs):
    def get_static_params():
        pass
    def get_static_hyperparams():
        pass
    def get_static_arch():
        pass
    def get_build_fn():
        pass

class SklearnKerasClassifier(KerasClassifier, GetStaticModel):
        def __init__(self, build_fn, **kwargs):
            super(KerasClassifier, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.best = kwargs['best']
            print("self.params from __init__ -- ", self.params)

        def fit(self, x_train, y_train, **kwargs):
            import numpy as np
            kwargs.setdefault('verbose', 0)
            verbose = kwargs['verbose']
            kwargs.setdefault('params', self.params)
            kwargs.setdefault('space', False)
            kwargs.setdefault('cuts_per_feature', False)

            y_train = np.array(y_train) # Compatibility with all formats?
            if len(y_train.shape) == 2 and y_train.shape[1] > 1:
                self.classes_ = np.arange(y_train.shape[1])
            elif (len(y_train.shape) == 2 and y_train.shape[1] == 1) or len(y_train.shape) == 1:
                self.classes_ = np.unique(y_train)
                y_train = np.searchsorted(self.classes_, y_train)
            else:
                raise ValueError(
                        'Invalid shape for y_train: ' + str(y_train.shape))

            primal_model = self.primal
            primal_model.fit(x_train, y_train)
            if primal_model.__class__.__name__ is not 'DecisionTreeClassifier':
                self.model = self.build_fn.__call__(x_train=x_train) # Not required when Tune is default.
                y_pred = primal_model.predict(x_train)

            else:
                if not kwargs['cuts_per_feature']:
                    feature_index, count = np.unique(primal_model.tree_.feature, return_counts=True)
                    cuts_per_feature = np.zeros(shape=x_train.shape[1], dtype=int)

                    for i, _ in enumerate(cuts_per_feature):
                        for j, value in enumerate(feature_index):
                            if i==value:
                                cuts_per_feature[i] = count[j]

                    cuts_per_feature = list(cuts_per_feature)

                    print("From wrappers -- ", cuts_per_feature)


                else:
                    cuts_per_feature = kwargs['cuts_per_feature']

                units = y_train.shape[1]
                self.model = self.build_fn.__call__(x_train=x_train, cuts_per_feature=cuts_per_feature,
                                                    units=units) # Not required when Tune is default.
            
            # Check whether to compute for the best model or not
            print("self.params from fit -- ", self.params)
            if (kwargs['params'] != self.params):
                # Optimize
                hyperopt_space = kwargs['space']
                # self.params.update(kwargs['params'])
                y_pred = primal_model.predict(x_train)
                primal_data = {
                    'y_pred': y_pred,
                    'model_name': primal_model.__class__.__name__
                }

                ## Search for best model using Tune ##
                if primal_model.__class__.__name__ is 'DecisionTreeClassifier':
                    self.model = get_best_model(x_train, y_train,
                                                primal_data=primal_data, params=kwargs['params'], 
                                                space=hyperopt_space, cuts_per_feature=cuts_per_feature,
                                                units=units, build_fn=self.build_fn)
                else:
                     self.model = get_best_model(x_train, y_train,
                                                primal_data=primal_data, params=kwargs['params'], 
                                                space=hyperopt_space, build_fn=self.build_fn)                   
                # self.model.fit(x_train, y_train, epochs=200,
                #             batch_size=30, verbose=verbose) # Not necessary. Fitting twice by now.
                return self.model # Not necessary.
            else:
                # Dont Optmize
                self.model.fit(x_train, y_train, epochs=500, batch_size=500, verbose=verbose)
                return self.model # Not necessary.

        def save(self, filename = None):
            if filename == None:
                raise ValueError('Name Error: to save the model you need to specify the filename')

            pickle.dump(self.model, open(filename + '.pkl', 'wb'))

            self.model.save(filename + '.h5')

            onnx_model = onnxmltools.convert_keras(self.model)
            onnxmltools.utils.save_model(onnx_model, filename + '.onnx')


class SklearnKerasRegressor(KerasRegressor):
        def __init__(self, build_fn, **kwargs):
            super(KerasRegressor, self).__init__(build_fn=build_fn)
            self.primal = kwargs['primal']
            self.params = kwargs['params']
            self.best = kwargs['best']

        def fit(self, x_train, y_train, **kwargs):
            # Check whether to compute for the best model or not
            kwargs.setdefault('verbose', 0)
            verbose = kwargs['verbose']
            if (self.best and self.params != None):
                # Optimize
                primal_model = self.primal
                primal_model.fit(x_train, y_train)
                y_pred = primal_model.predict(x_train)
                primal_data = {
                    'y_pred': y_pred,
                    'model_name': primal_model.__class__.__name__
                }

                self.model, final_epoch, final_batch_size = get_best_model( # Fix! Returns only `model`
                    x_train, y_train, primal_data=primal_data,
                    params=self.params, build_fn=self.build_fn)
                # Epochs and batch_size passed in Talos as well
                self.model.fit(x_train, y_train, epochs=final_epoch,
                    batch_size=final_batch_size, verbose=verbose)
                return self.model
            else:
                # Dont Optmize
                self.model = self.build_fn.__call__(x_train=x_train)
                self.model.fit(x_train, y_train, epochs=500, batch_size=500)
                return self.model

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

wrappers = {
    'LogisticRegression': SklearnKerasClassifier,
    'LinearRegression': SklearnKerasRegressor,
    'LinearDiscriminantAnalysis': SklearnKerasClassifier,
    'DecisionTreeClassifier': SklearnKerasClassifier
}
