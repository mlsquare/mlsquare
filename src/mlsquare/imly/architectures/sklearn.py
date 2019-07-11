#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.regularizers import l1_l2
import numpy as np
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from ..base import registry, BaseModel
from ..wrappers.sklearn import SklearnKerasClassifier, SklearnKerasRegressor
from ..utils.layers import DecisionTree
from ..utils.functions import _parse_params


class GeneralizedLinearModel(BaseModel):  # Rename glm

    def create_model(self):

        model_params = _parse_params(self._model_params, return_as='nested')
        # Why make it private? Alternate name?
        # Move parsing to base model
        model = Sequential()
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            units = 1
        else:
            units = self.y.shape[1]
        model_params['layer_1'].update({'input_dim': self.X.shape[1], 'units': units})
        ## Update the units choice before moving forward
        model.add(Dense(units=model_params['layer_1']['units'],
                        input_dim=model_params['layer_1']['input_dim'],
                        activation=model_params['layer_1']['activation'],
                        kernel_regularizer=l1_l2(l1=model_params['layer_1']['l1'],
                                                 l2=model_params['layer_1']['l2'])))  # Update as l1() for lasso
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model

    def set_params(self, **kwargs):
        kwargs.setdefault('params', None)
        kwargs.setdefault('set_by', None)
        if kwargs['set_by'] == 'model_init':
            self._model_params = _parse_params(kwargs['params'], return_as='flat')
        elif kwargs['set_by'] == 'opitmizer':
            self._model_params = kwargs['params']
        else:
            self._model_params = kwargs['params']

    def get_params(self):
        return self._model_params

    def update_params(self, params):
        # Should this be a special case of set_params? Wouldn't that be clutter
        self._model_params.update(params)

    def wrapper(self):
        return self._wrapper


@registry.register
class LogisticRegression(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasClassifier
        self.module_name = 'sklearn'
        self.model_name = 'LogisticRegression'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1,
                                    'l1': 0,
                                    'l2': 0,
                                    'activation': 'sigmoid'},
                        'optimizer': 'adam',
                        'loss': 'binary_crossentropy'
                        }

        # Reduntant? Can this be instead handled by update method alone?
        self.set_params(params=model_params, set_by='model_init')


@registry.register
class LinearRegression(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasRegressor
        self.module_name = 'sklearn'
        model_params = {'units': 1,
                        'l1': 0,
                        'l2': 0,
                        'activation': 'linear',
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        # self.set_params(model_params)
# Yet to test


@registry.register
class Ridge(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasRegressor
        self.module_name = 'sklearn'
        model_params = {'units': 1,
                        'l1': 0,
                        'l2': 0.1,  # Should be configurable at tune level. Dependant on input
                        'activation': 'linear',
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        # self.set_params(model_params)
# Yet to test


@registry.register
class Lasso(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasRegressor
        self.module_name = 'sklearn'
        model_params = {'units': 1,
                        'l1': 0.1,
                        'l2': 0,
                        'activation': 'linear',
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        # self.set_params(model_params)
# Yet to test


@registry.register
class ElasticNet(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasRegressor
        self.module_name = 'sklearn'
        model_params = {'units': 1,
                        'l1': 0.1,
                        'l2': 0.1,
                        'activation': 'linear',
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        # self.set_params(model_params)


@registry.register
class LinearSVC(GeneralizedLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasClassifier
        self.module_name = 'sklearn'
        model_params = {  # 'units': 2, # Update
            'l1': 0,
            'l2': 0,
            'activation': 'linear',
            'optimizer': 'adam',
            'loss': 'categorical_hinge'
        }

        # self.set_params(model_params)

    def transform_data(self, X, y, y_pred):  # Move this to wrapper
        if len(y.shape) == 1:  # Test with multiple target shapes
            y = y.reshape(-1, 1)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y)
        y_pred = self.enc.transform(y_pred.reshape([-1, 1]))
        return X, y, y_pred


class KernelGeneralizedLinearModel(GeneralizedLinearModel):
    def create_model(self, **kwargs):
        # X = kwargs['X']
        # y = kwargs['y']
        # self.update_params({'layer_2': {'units': y.shape[1]}}) ## Moved from tune to model
        model_params = self._model_params
        model = Sequential()

        model.add(Dense(units=model_params['layer_1']['kernel_dim'],
                        trainable=False, kernel_initializer='random_normal', ## Connect with sklearn_config
                        activation=model_params['layer_1']['activation']))
        model.add(Dense(model_params['layer_2']['units'],  # Change to len(target.shape[1])
                        activation=model_params['layer_2']['activation']))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])
        # model.add(Dense(model_params['layer_1'])) -- Check if keras accepts dict as params
        

        # model.add(Dense(units=10,
        #                 trainable=False, kernel_initializer='random_normal',
        #                 activation='linear'))
        # model.add(Dense(2,  # Change to len(target.shape[1])
        #                 activation='softmax'))
        # model.compile(optimizer='adam',
        #               loss='categorical_hinge',
        #               metrics=['accuracy'])

        return model


@registry.register
class SVC(KernelGeneralizedLinearModel):
    def __init__(self):
        self.module_name = 'sklearn'
        model_params = {'layer_1': {'kernel_dim': 10, ## Make it 'units'
                                    'activation': 'linear'
                                    },
                        'layer_2': { 'units': 2,
                                    'activation': 'softmax'
                                    },
                        'optimizer': 'adam',
                        'loss': 'categorical_hinge'}
                        # {'layer_1_activation': 'linear'}
        
        ## Flatten the params dict at tune level
        self.wrapper = SklearnKerasClassifier
        # self.set_params(model_params)

    def transform_data(self, X, y, y_pred):
        if len(y.shape) == 1:
            y = y.reshape(-1,1)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y)
        y_pred = np.array(self.enc.transform(y_pred.reshape([-1, 1])))
        return X, y, y_pred

# @registry.register
# class SVC(KernelGeneralizedLinearModel):
#     def __init__(self):
#         self.module_name = 'sklearn'
#         model_params = {'kernel_dim': 10,
#                         'activation': 'linear'
#                         'units': 2,
#                         'activation': 'softmax'
                                    
#                         'optimizer': 'adam',
#                         'loss': 'categorical_hinge'}
#         self.wrapper = SklearnKerasClassifier
#         self.set_params(model_params)

#     def transform_data(self, X, y, y_pred):
#         if len(y.shape) == 1:
#             y = y.reshape(-1,1)
#         self.enc = OneHotEncoder(handle_unknown='ignore')
#         self.enc.fit(y)
#         y_pred = self.enc.transform(y_pred.reshape([-1, 1]))
#         return X, y, y_pred


class CART(GeneralizedLinearModel):
    def __init__(self):
        self.X = None
        self.y = None
        self.primal = None
        self.cuts_per_feature = None
        self.wrapper = SklearnKerasClassifier
        self.module_name = 'sklearn'
        model_params = {
            'activation': 'sigmoid',
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy'
        }

        # self.set_params(model_params)

    def create_model(self, **kwargs):
        cuts_per_feature = self.cuts_per_feature
        if cuts_per_feature is None:
            feature_index, count = np.unique(self.primal.tree_.feature, return_counts=True)
            cuts_per_feature = np.zeros(shape=self.X.shape[1], dtype=int)

            for i, _ in enumerate(cuts_per_feature):
                for j, value in enumerate(feature_index):
                    if i==value:
                        cuts_per_feature[i] = count[j]

            cuts_per_feature = list(cuts_per_feature)

        else:
            cuts_per_feature = self.cuts_per_feature



        model_params = self._model_params
        # cuts_per_feature = kwargs['model_params']['cuts_per_feature']
        # Update to kwargs['model_params']['cuts_per_feature']
        if type(cuts_per_feature) not in (list, int):
            raise TypeError(
                'cuts_per_feature should be of type `list` or `int`')
        elif type(cuts_per_feature) is int:
            if cuts_per_feature > np.ceil(self.X.shape[0]):
                cuts_per_feature = [np.ceil(self.X.shape[0]) for i in range(
                    self.X.shape[1])]
            else:
                cuts_per_feature = [cuts_per_feature for i in range(
                    self.X.shape[1])]
        else:
            if len(cuts_per_feature) != self.X.shape[1]:
                print("From arch -- ", cuts_per_feature)
                raise ValueError(
                    'The length of `cuts_per_feature` should be equal to number of features.')
            else:
                cuts_per_feature = [np.ceil(self.X.shape[0])
                                    if i > np.ceil(self.X.shape[0]) else i for i in cuts_per_feature]
        # kwargs.setdefault('units', kwargs['model_params']['units'])
        # visible = Input(shape=(kwargs['x_train'].shape[1],))
        visible = Input(shape=(self.X.shape[1],)) ## kwargs or self
        # hidden = DecisionTree(cuts_per_feature=kwargs['cuts_per_feature'])(visible)
        hidden = DecisionTree(cuts_per_feature=cuts_per_feature)(visible)
        output = Dense(self.y.shape[1],
                       activation=model_params['activation'])(hidden)
        model = Model(inputs=visible, outputs=output)

        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model

@registry.register
class DecisionTreeClassifier(CART):
    ''' '''


def linear_discriminant_analysis(**kwargs):  # Refactor!
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.regularizers import l2
        from ..utils.losses import mse_in_theano

        model = Sequential()
        model.add(Dense(
            kwargs['params']['units'],
            input_dim=kwargs['x_train'].shape[1],
            activation=kwargs['params']['activation'][0],
            kernel_regularizer=l2(1e-5)
        ))
        model.compile(
            optimizer=kwargs['params']['optimizer'],
            loss=mse_in_theano,
            metrics=['accuracy']
        )

        return model
    except ImportError:
        print("keras is required to transpile the model")
        return False


def cart(**kwargs):
    try:
        from keras.models import Model
        from keras.layers import Input, Dense
        import numpy as np
        from ..utils.layers import DecisionTree

        # This validation is not moved to the DT Layer since x_train.shape[0] is not
        # available at that level.

        cuts_per_feature = kwargs['model_params']['cuts_per_feature']
        # Update to kwargs['model_params']['cuts_per_feature']
        if type(cuts_per_feature) not in (list, int):
            raise TypeError(
                'cuts_per_feature should be of type `list` or `int`')
        elif type(cuts_per_feature) is int:
            if cuts_per_feature > np.ceil(kwargs['x_train'].shape[0]):
                cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0]) for i in range(
                    kwargs['x_train'].shape[1])]
            else:
                cuts_per_feature = [cuts_per_feature for i in range(
                    kwargs['x_train'].shape[1])]
        else:
            if len(cuts_per_feature) != kwargs['x_train'].shape[1]:
                print("From arch -- ", cuts_per_feature)
                raise ValueError(
                    'The length of `cuts_per_feature` should be equal to number of features.')
            else:
                cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0])
                                    if i > np.ceil(kwargs['x_train'].shape[0]) else i for i in cuts_per_feature]

        model_params = kwargs['model_params']
        kwargs.setdefault('units', kwargs['model_params']['units'])
        visible = Input(shape=(kwargs['x_train'].shape[1],))
        # hidden = DecisionTree(cuts_per_feature=kwargs['cuts_per_feature'])(visible)
        hidden = DecisionTree(cuts_per_feature=cuts_per_feature)(visible)
        output = Dense(kwargs['units'],
                       activation=model_params['activation'])(hidden)
        model = Model(inputs=visible, outputs=output)

        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print("keras is required to transpile the model")
        # Raise error instead of returning False. False doesn't help much while debugging.
        return False


def kernel_glm(**kwargs):  # Update in config
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense

        # Temporary hack. Needs to be fixed during architectural refactoring.
        kwargs.setdefault('y_train', None)

        model_params = kwargs['model_params']
        model = Sequential()
        model.add(Dense(kwargs['y_train'].shape[1],
                        input_dim=kwargs['x_train'].shape[1],
                        activation=model_params['activation']))
        model.add(Dense(model_params['kernel_dim'],  # kernel dimensions - hyperparams
                        # Check for random_normal - for rbf
                        trainable=False, kernel_initializer='random_normal',
                        activation=model_params['activation']))
        model.add(Dense(kwargs['y_train'].shape[1],
                        activation='softmax'))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print("keras is required to transpile the model")
        return False


'''
Qs
1) Data transformation level - model, commons + wrapper or external?
  Use OHE for every classification algorithms
[X] 2) update_params - should it be a part of base model?
3) Change the structure of model_params. Add options for layer level details
4) Update 'units' and 'input_dim' within model vs from tune
5) Keras custom name for layers - especially for functional API
Come up with a naming convention for keras layers. Check existing keras convention

6) {'layer_1_units': 10,
    'layer_2_units': 12}
Parse this back at create_model().
7) CART
    + Consider dt_regression as well.

Updates(09/07/2019) - 
1) Pass all meaningful kwargs from 'wrapper.fit()' to the model - primal, X, y, cuts_per_feature etc..
These are then expected to be validated in each model's __init__.
Data preperation needs to be done by the wrapper.

Check SOLID principles to clarify the confusion on transform method and data flow issues.

Notes on multilayer changes - 
1) User provides dict in nested structure.
2) Flatten it at set_params.
3) Convert back to nested structure at create_model level.
4) Why not follow a flattened approach throughout the flow - arch + tune + set/create methods

Checklist (validate each changes on each algorithm) - 
1) Logistic regression [multilayer_params[DONE], , OHE[], transform_data[]]
'''
