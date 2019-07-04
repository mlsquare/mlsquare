# #!/usr/bin/env python
# # -*- coding: utf-8 -*-

# from keras.wrappers.scikit_learn import KerasRegressor, KerasClassifier
# from ..optmizers import get_best_model
# import pickle
# import onnxmltools
# from numpy import where
# from sklearn.preprocessing import OneHotEncoder

# class SklearnKerasClassifier(KerasClassifier):
#     def __init__(self, build_fn, **kwargs):
#         super(KerasClassifier, self).__init__(build_fn=build_fn)
#         self.primal = kwargs['primal']
#         self.params = kwargs['params']
#         self.best = kwargs['best']
#         self.enc = None

#     def fit(self, x_train, y_train, **kwargs):
#         import numpy as np
#         kwargs.setdefault('verbose', 0)
#         verbose = kwargs['verbose']
#         kwargs.setdefault('params', self.params)
#         # Update as kwargs.setdefault('params', {})
#         kwargs.setdefault('space', False)
#         kwargs.setdefault('cuts_per_feature', False)

#         y_train = np.array(y_train)  # Compatibility with all formats?
#         if len(y_train.shape) == 2 and y_train.shape[1] > 1:
#             self.classes_ = np.arange(y_train.shape[1])
#         elif (len(y_train.shape) == 2 and y_train.shape[1] == 1) or len(y_train.shape) == 1:
#             self.classes_ = np.unique(y_train)
#             y_train = np.searchsorted(self.classes_, y_train)
#         else:
#             raise ValueError(
#                 'Invalid shape for y_train: ' + str(y_train.shape))

#         primal_model = self.primal
#         primal_model.fit(x_train, y_train)
#         y_pred = primal_model.predict(x_train)      
#         # This check is temporary. This will be moved to 'AbstractModelClass' after
#         # the Architectural refactoring is done.
#         if primal_model.__class__.__name__ in ('LinearSVC', 'SVC'):
#             self.enc = OneHotEncoder(handle_unknown='ignore')
#             self.enc.fit(y_train)
#             y_pred = self.enc.transform(y_pred.reshape([-1, 1]))
 
#         elif primal_model.__class__.__name__ is 'DecisionTreeClassifier':
#             if not kwargs['cuts_per_feature']:
#                 feature_index, count = np.unique(primal_model.tree_.feature, return_counts=True)
#                 cuts_per_feature = np.zeros(shape=x_train.shape[1], dtype=int)

#                 for i, _ in enumerate(cuts_per_feature):
#                     for j, value in enumerate(feature_index):
#                         if i==value:
#                             cuts_per_feature[i] = count[j]

#                 cuts_per_feature = list(cuts_per_feature)

#             else:
#                 cuts_per_feature = kwargs['cuts_per_feature']

#             units = y_train.shape[1]
#             # This is assuming that Tune must run by default. Currently, user will have to send 
#             # 'params' to activate Tune.
#             kwargs['params'].update({'units': units, 'cuts_per_feature': cuts_per_feature})
#             # y_pred = y_train

#         primal_data = {
#             'y_pred': y_pred,
#             'model_name': primal_model.__class__.__name__
#         }


#         # Check whether to compute for the best model or not
#         if (kwargs['params'] != self.params):
#             # Optimize
#             hyperopt_space = kwargs['space']
#             # self.params.update(kwargs['params'])

#             ## Search for best model using Tune ##
#             self.model = get_best_model(x_train, y_train, build_fn = self.build_fn,
#                                         primal_data=primal_data, params=kwargs['params'],
#                                         space=hyperopt_space)
#             return self.model  # Not necessary.
#         else:
#             # Dont Optmize
#             self.model = self.build_fn.__call__(x_train=x_train)
#             self.model.fit(x_train, y_pred, epochs=500,
#                            batch_size=500, verbose=verbose)
#             return self.model  # Not necessary.

#     def save(self, filename=None):
#         if filename == None:
#             raise ValueError(
#                 'Name Error: to save the model you need to specify the filename')

#         pickle.dump(self.model, open(filename + '.pkl', 'wb'))

#         self.model.save(filename + '.h5')

#         onnx_model = onnxmltools.convert_keras(self.model)
#         onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

#     ## Temporary hack. This should be handled during architectural refactoring.
#     def score(self, x, y, **kwargs):
#         if self.enc is not None:
#             y = self.enc.transform(y)
#             self.y_out = y

#         score = self.model.evaluate(x, y, **kwargs)
#         return score

#     def explain(self):
#         # @param: SHAP or interpret
#         pass



# class SklearnKerasRegressor(KerasRegressor):
#     def __init__(self, build_fn, **kwargs):
#         super(KerasRegressor, self).__init__(build_fn=build_fn)
#         self.primal = kwargs['primal']
#         self.params = kwargs['params']
#         self.best = kwargs['best']

#     def fit(self, x_train, y_train, **kwargs):
#         # Check whether to compute for the best model or not
#         kwargs.setdefault('verbose', 0)
#         verbose = kwargs['verbose']
#         if (kwargs['params'] != self.params):
#             # Optimize
#             primal_model = self.primal
#             primal_model.fit(x_train, y_train)
#             y_pred = primal_model.predict(x_train)
#             primal_data = {
#                 'y_pred': y_pred,
#                 'model_name': primal_model.__class__.__name__
#             }

#             self.model = get_best_model(x_train, y_train, primal_data=primal_data,
#                                         params=kwargs['params'], build_fn=self.build_fn)
#             return self.model
#         else:
#             # Dont Optmize
#             self.model = self.build_fn.__call__(x_train=x_train)
#             self.model.fit(x_train, y_train, epochs=500, batch_size=500)
#             return self.model

#     def score(self, x, y, **kwargs):
#         score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
#         # keras_regressor treats all score values as loss and adds a '-ve' before passing
#         return -score

#     def save(self, filename=None):
#         if filename == None:
#             raise ValueError(
#                 'Name Error: to save the model you need to specify the filename')

#         pickle.dump(self.model, open(filename + '.pkl', 'wb'))

#         self.model.save(filename + '.h5')

#         onnx_model = onnxmltools.convert_keras(self.model)
#         onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

# wrappers = {
#     'LogisticRegression': SklearnKerasClassifier,
#     'LinearRegression': SklearnKerasRegressor,
#     'LinearDiscriminantAnalysis': SklearnKerasClassifier,
#     'DecisionTreeClassifier': SklearnKerasClassifier,
#     'LinearSVC': SklearnKerasClassifier,
#     'SVC': SklearnKerasClassifier,
#     'Ridge': SklearnKerasRegressor,
#     'Lasso': SklearnKerasRegressor,
#     'ElasticNet': SklearnKerasRegressor
# }

## #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### #### ##


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
        super(KerasClassifier, self).__init__(build_fn=abstract_model.create_model())
        self.primal = primal
        self.params = None ## Temporary!
        self.abstract_model = abstract_model
        # self.params = kwargs['params']
        # self.best = kwargs['best']
        # self.enc = None

    def fit(self, X, y, **kwargs): # Why kwargs?
        import numpy as np
        kwargs.setdefault('verbose', 0)
        verbose = kwargs['verbose']
        kwargs.setdefault('params', self.params)
        # Update as kwargs.setdefault('params', {})
        kwargs.setdefault('space', False)
        # kwargs.setdefault('cuts_per_feature', False) # Move to dt class

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
        # This check is temporary. This will be moved to 'AbstractModelClass' after
        # the Architectural refactoring is done.
        ## Move to BaseClass
        # if primal_model.__class__.__name__ in ('LinearSVC', 'SVC'):
        #     self.enc = OneHotEncoder(handle_unknown='ignore')
        #     self.enc.fit(y)
        #     y_pred = self.enc.transform(y_pred.reshape([-1, 1]))
 
        # elif primal_model.__class__.__name__ is 'DecisionTreeClassifier':
        #     if not kwargs['cuts_per_feature']:
        #         feature_index, count = np.unique(primal_model.tree_.feature, return_counts=True)
        #         cuts_per_feature = np.zeros(shape=X.shape[1], dtype=int)

        #         for i, _ in enumerate(cuts_per_feature):
        #             for j, value in enumerate(feature_index):
        #                 if i==value:
        #                     cuts_per_feature[i] = count[j]

        #         cuts_per_feature = list(cuts_per_feature)

        #     else:
        #         cuts_per_feature = kwargs['cuts_per_feature']

        #     units = y.shape[1]
        #     # This is assuming that Tune must run by default. Currently, user will have to send 
        #     # 'params' to activate Tune.
        #     kwargs['params'].update({'units': units, 'cuts_per_feature': cuts_per_feature})
        #     # y_pred = y_train

        primal_data = { ## Consider renaming -- primal_model_data or primal_results
            'y_pred': y_pred,
            'model_name': primal_model.__class__.__name__
        }


        # Check whether to compute for the best model or not
        # if (kwargs['params'] != self.params):
            # Optimize
        # hyperopt_space = kwargs['space']
        # self.params.update(kwargs['params'])

        ## Search for best model using Tune ##
        self.final_model = get_best_model(X, y, abstract_model = self.abstract_model, primal_data=primal_data)
        return self.final_model  # Not necessary.
        # else:
        #     # Dont Optmize
        #     self.model = self.build_fn.__call__(X=X)
        #     self.model.fit(X, y_pred, epochs=500,
        #                    batch_size=500, verbose=verbose)
        #     return self.model  # Not necessary.

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
        # if self.enc is not None:
        #     y = self.enc.transform(y)
        #     self.y_out = y

        score = self.final_model.evaluate(x, y, **kwargs)
        return score

    def explain(self):
        # @param: SHAP or interpret
        pass



# class SklearnKerasRegressor(KerasRegressor):
#     def __init__(self, build_fn, **kwargs):
#         super(KerasRegressor, self).__init__(build_fn=build_fn)
#         self.primal = kwargs['primal']
#         self.params = kwargs['params']
#         self.best = kwargs['best']

#     def fit(self, x_train, y_train, **kwargs):
#         # Check whether to compute for the best model or not
#         kwargs.setdefault('verbose', 0)
#         verbose = kwargs['verbose']
#         if (kwargs['params'] != self.params):
#             # Optimize
#             primal_model = self.primal
#             primal_model.fit(x_train, y_train)
#             y_pred = primal_model.predict(x_train)
#             primal_data = {
#                 'y_pred': y_pred,
#                 'model_name': primal_model.__class__.__name__
#             }

#             self.model = get_best_model(x_train, y_train, primal_data=primal_data,
#                                         params=kwargs['params'], build_fn=self.build_fn)
#             return self.model
#         else:
#             # Dont Optmize
#             self.model = self.build_fn.__call__(x_train=x_train)
#             self.model.fit(x_train, y_train, epochs=500, batch_size=500)
#             return self.model

#     def score(self, x, y, **kwargs):
#         score = super(SklearnKerasRegressor, self).score(x, y, **kwargs)
#         # keras_regressor treats all score values as loss and adds a '-ve' before passing
#         return -score

#     def save(self, filename=None):
#         if filename == None:
#             raise ValueError(
#                 'Name Error: to save the model you need to specify the filename')

#         pickle.dump(self.model, open(filename + '.pkl', 'wb'))

#         self.model.save(filename + '.h5')

#         onnx_model = onnxmltools.convert_keras(self.model)
#         onnxmltools.utils.save_model(onnx_model, filename + '.onnx')

# wrappers = {
#     'LogisticRegression': SklearnKerasClassifier,
#     'LinearRegression': SklearnKerasRegressor,
#     'LinearDiscriminantAnalysis': SklearnKerasClassifier,
#     'DecisionTreeClassifier': SklearnKerasClassifier,
#     'LinearSVC': SklearnKerasClassifier,
#     'SVC': SklearnKerasClassifier,
#     'Ridge': SklearnKerasRegressor,
#     'Lasso': SklearnKerasRegressor,
#     'ElasticNet': SklearnKerasRegressor
# }