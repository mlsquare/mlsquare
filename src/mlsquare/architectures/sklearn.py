#!/usr/bin/env python
# -*- coding: utf-8 -*-
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.regularizers import l1_l2
import numpy as np
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from ..base import registry, BaseModel, BaseTransformer
from ..adapters.sklearn import SklearnKerasClassifier, SklearnKerasRegressor, SklearnTfTransformer, SklearnPytorchClassifier, IrtKerasRegressor
from ..layers.keras import DecisionTree
from ..utils.functions import _parse_params
import tensorflow as tf
import pandas
from abc import abstractmethod
# from ..losses import lda_loss

import keras
from keras import backend as K
from keras.layers import Lambda
from keras import regularizers
from keras import initializers
from keras.layers import Activation
from keras import metrics

class GeneralisedIrtModel(BaseModel):
    def create_model(self, **kwargs):
        model_params = _parse_params(self._model_params, return_as='nested')
        model_params.update({'input_dims_users':self.x_train_user.shape[1],'input_dims_items':self.x_train_questions.shape[1]})
        
        user_input_layer = Input(shape=(model_params['input_dims_users'],), name= 'user_id')#top half of input
        quest_input_layer = Input(shape=(model_params['input_dims_items'],), name='questions/items')#bottom half of input
        
        if not self.l_traits==None:
            pass#provision to add latent traits to first layer
            #latent_trait = Dense(1, use_bias=False,
            #    kernel_initializer= initializers.RandomNormal(mean=0, stddev=1.0, seed=None),
            #    kernel_regularizer=regularizers.l2(0.01), name='latent_trait')(user_input_layer)
        else:
            latent_trait = Dense(model_params['ability_params']['units'], use_bias=False,
                kernel_initializer= model_params['ability_params']['kernel_init'],# initializers.RandomNormal(mean=model_params['ability_params']['initializers']['dist']['normal']['mean'],
                #stddev=model_params['ability_params']['initializers']['dist']['normal']['stddev'], seed=None),
                kernel_regularizer=l1_l2(l1=model_params['regularizers']['l1'], l2= model_params['regularizers']['l2']),
                name='latent_trait')(user_input_layer)

        #2. kernel init set to RandomNorm(0,1)
        #b_j
        difficulty_level = Dense(model_params['ability_params']['units'], use_bias=False,
            kernel_initializer= model_params['diff_params']['kernel_init'],#initializers.RandomNormal(mean=model_params['diff_params']['initializers']['dist']['normal']['mean'],
            #stddev=model_params['diff_params']['initializers']['dist']['normal']['stddev'], seed=None),
            name='difficulty_level')(quest_input_layer)#b_j

        #3. kernel init set to RandomNorm(1,1)
        # Descrimination- also lamda_j
        discrimination_param = Dense(model_params['ability_params']['units'], use_bias=False,
            kernel_initializer= model_params['disc_params']['kernel_init'],#initializers.RandomNormal(mean=model_params['disc_params']['initializers']['dist']['normal']['mean'],
            #stddev=model_params['disc_params']['initializers']['dist']['normal']['stddev'], seed=None),
            trainable=model_params['disc_params']['train'],
            activation= model_params['disc_params']['act'],#kernel_constraint= constraints.NonNeg(),
            name='disc_param')(quest_input_layer)

        #lamda_j*t_i
        disc_latent_interaction = keras.layers.Multiply(name='lambda_latent_inter.')([discrimination_param, latent_trait])

        #alpha_j(= lambda_j*b_j)
        disc_diff_interaction = keras.layers.Multiply(name='alpha_param.')([discrimination_param, difficulty_level])

        #alpha_j + lamda_j*t_i]
        alpha_lambda_add = keras.layers.Subtract(name='alpha_lambda_add')([disc_latent_interaction, disc_diff_interaction])# -alpha+lambda*latent_traits

        #Sigmoid[alpha_j + lamda_j*t_i]
        sigmoid_layer= Activation('sigmoid', name='Sigmoid_func')(alpha_lambda_add)

        #c_j
        guess_param = Dense(model_params['ability_params']['units'], use_bias=False,
            kernel_initializer= model_params['guess_params']['kernel_init'],#initializers.RandomUniform(minval=model_params['guess_params']['initializers']['dist']['uniform']['minval'],
            #maxval=model_params['guess_params']['initializers']['dist']['uniform']['maxval'], seed=None), trainable=model_params['guess_params']['train'],
            activation=model_params['guess_params']['act'], name='guessing_param')(quest_input_layer)

        #5. Sigmoid positioning corrected as per 3PL expression
        #(1-c_j)
        guess_param_interaction= Lambda(lambda x: K.constant(value=np.array([1 -model_params['guess_params']['slip']])) - x, name='guess_param_inter.')(guess_param)

        #(1-c_j)*sigmoid[]
        guess_param_interaction= keras.layers.Multiply(name='disc/guess_param_inter.')([sigmoid_layer, guess_param_interaction])
        #c_j+ (1-c_j)*sigmoid[]
        guess_param_interaction= keras.layers.Add(name='guess_param_inter/add')([guess_param, guess_param_interaction])


        #6. changed activation to just linear
        prediction_output = Dense(model_params['hyper_params']['units'], trainable=False, use_bias=False,
            kernel_initializer=keras.initializers.Ones(), name='prediction_layer')(guess_param_interaction)


        model_ = Model(inputs=[user_input_layer, quest_input_layer], outputs= prediction_output)
        model_.compile(loss= model_params['hyper_params']['loss'], optimizer=model_params['hyper_params']['optimizer'], metrics= ['mae', 'accuracy'])
        return model_

    def set_params(self, **kwargs):
        kwargs.setdefault('params', None)
        kwargs.setdefault('set_by', None)
        if kwargs['set_by'] == 'model_init':
            ## Detect nested or flat at parse_params level
            self._model_params = _parse_params(kwargs['params'], return_as='flat')
        elif kwargs['set_by'] == 'opitmizer':
            self._model_params = kwargs['params']
        else:
            self._model_params = kwargs['params']

    def get_params(self):
        return self._model_params

    def update_params(self, params):
        self._model_params.update(params)
        self.get_initializers(self._model_params)

    def adapter(self):
        return self._adapter

    def get_initializers(self, params):
        default_params={'backend':'keras', 'distrib':'normal', 'mean':0, 'stddev':1,'minval':0, 'maxval':0}
        for key, vals in params.items():
            sub_dict= default_params.copy()
            if 'init_params' in vals.keys():
                sub_dict.update(params[key]['init_params'])
                
                if sub_dict['backend']=='keras' and sub_dict['distrib']=='normal':
                    params[key].update({'kernel_init':initializers.RandomNormal(mean=sub_dict['mean'], stddev=sub_dict['stddev'])})
                else:# sub_dict['backend']=='keras' and sub_dict['distrib']=='uniform'
                    params[key].update({'kernel_init':initializers.RandomUniform(minval=sub_dict['minval'], maxval=sub_dict['maxval'])})
        self._model_params.update(params)



    
@registry.register
class KerasIrt1PLModel(GeneralisedIrtModel):
    def __init__(self):
        self.adapter = IrtKerasRegressor
        self.module_name= 'mlsquare'#'embibe'
        self.name= 'rasch'
        self.version= 'default'
        model_params = {'ability_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'diff_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'disc_params':{'units':1, 'init_params':{'stddev':0},'train':False, 'act':'exponential'},#default 'keras','normal'
                        'guess_params':{'units':1, 'init_params':{'distrib':'uniform'}, 'train':False, 'act':'linear', 'slip':0},#default 'keras','uniform'
                        'regularizers':{'l1':0, 'l2':0},
                        'hyper_params':{'units':1, 'optimizer': 'sgd', 'loss': 'binary_crossentropy'}}
        
        self.set_params(params=model_params, set_by='model_init')
        self.update_params(model_params)

@registry.register
class KerasIrt2PLModel(GeneralisedIrtModel):
    def __init__(self):
        self.adapter = IrtKerasRegressor
        self.module_name= 'mlsquare'#'embibe'
        self.name= 'twoPl'
        self.version= 'default'#'2PL'

        model_params = {'ability_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'diff_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'disc_params':{'units':1, 'init_params':{},'train':True, 'act':'exponential'},#default 'keras','normal'
                        'guess_params':{'units':1, 'init_params':{'distrib':'uniform'}, 'train':False, 'act':'linear', 'slip':0},#default 'keras','uniform'
                        'regularizers':{'l1':0, 'l2':0},
                        'hyper_params':{'units':1, 'optimizer': 'sgd', 'loss': 'binary_crossentropy'}}

        self.set_params(params=model_params, set_by='model_init')
        self.update_params(model_params)

@registry.register
class KerasIrt3PLModel(GeneralisedIrtModel):
    def __init__(self):
        self.adapter = IrtKerasRegressor
        self.module_name= 'mlsquare'#'embibe'
        self.name= 'tpm'
        self.version= 'default'#'default'
        model_params = {'ability_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'diff_params':{'units':1, 'init_params':{}},#default 'keras','normal'
                        'disc_params':{'units':1, 'init_params':{},'train':True, 'act':'exponential'},#default 'keras','normal'
                        'guess_params':{'units':1, 'init_params':{'distrib':'uniform', 'minval':-3.5, 'maxval':-2.5}, 'train':True, 'act':'linear', 'slip':0},#default 'keras','uniform'
                        'regularizers':{'l1':0, 'l2':0},
                        'hyper_params':{'units':1, 'optimizer': 'sgd', 'loss': 'binary_crossentropy'}}

        self.set_params(params=model_params, set_by='model_init')
        self.update_params(model_params)


class GeneralizedLinearModel(BaseModel):
    """
	A base class for all generalized linear models.

    This class can be used as a base class for any glm models.
    While implementing ensure all required parameters are over written
    with the respective models values. Please refer to LogisticRegression or
    LinearRegression for implementation details.

    Methods
    -------
	create_model(model_params)
        This method contains the base template for glm models

	set_params(params)
        Method to set model parameters. This method handles the
        flattening of params as well.

	get_params()
        Method to read params.

	update_params(params)
        Method to update params.

    """

    def create_model(self, **kwargs):

        model_params = _parse_params(self._model_params, return_as='nested')
        # Why make it private? Alternate name?
        # Move parsing to base model
        model = Sequential()

        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
        ## Use OHE for all classification algorithms
        ## Check for class numbers in y
        ## Use that as units count
            units = 1
        else:
            units = self.y.shape[1]
        model_params['layer_1'].update({'input_dim': self.X.shape[1], 'units': units})
        model.add(Dense(units=model_params['layer_1']['units'],
                        input_dim=model_params['layer_1']['input_dim'],
                        activation=model_params['layer_1']['activation'],
                        kernel_regularizer=l1_l2(l1=model_params['layer_1']['l1'],
                                                 l2=model_params['layer_1']['l2'])))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model

    def set_params(self, **kwargs):
        kwargs.setdefault('params', None)
        kwargs.setdefault('set_by', None)
        if kwargs['set_by'] == 'model_init':
            ## Detect nested or flat at parse_params level
            self._model_params = _parse_params(kwargs['params'], return_as='flat')
        elif kwargs['set_by'] == 'opitmizer':
            self._model_params = kwargs['params']
        else:
            self._model_params = kwargs['params']

    def get_params(self):
        return self._model_params

    def update_params(self, params):
        self._model_params.update(params)

    def adapter(self):
        return self._adapter

class MatrixDecomposition(BaseTransformer):
    """
	A base class for all matrix decomposition models.

    This class can be used as a base class for matrix decomposition models.
    While implementing ensure all required methods are implemented or over written

    Methods
    -------
	set_params(params)
        Method to set model parameters. This method handles the
        flattening of params as well.

	get_params()
        Method to read params.

	update_params(params)
        Method to update params.
    """
    def fit(X, y, **kwargs):
        pass

    def fit_transform(X, y, **kwargs):
        pass

    def set_params(self, **kwargs):
        kwargs.setdefault('params', None)
        self._model_params = _parse_params(kwargs['params'], return_as='flat')

    def get_params(self):
        return self._model_params

    def update_params(self, params):
        self._model_params.update(params)

@registry.register
class SVD(MatrixDecomposition):
    def __init__(self):
        self.adapter = SklearnTfTransformer
        self.module_name = 'sklearn'
        self.name = 'TruncatedSVD'
        self.version = 'default'
        model_params = {'full_matrices': False, 'compute_uv': True, 'name':None}
        self.set_params(params=model_params)

    def fit(self, X, y=None,**kwargs):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None,**kwargs):
        model_params= _parse_params(self._model_params, return_as='flat')

        #changing to recommended dtype, accomodating dataframe & numpy array
        X = np.array(X, dtype= np.float32 if str(X.values.dtype)==
        'float32' else np.float64) if isinstance(X,
        pandas.core.frame.DataFrame) else np.array(X, dtype= np.float32
        if str(X.dtype)=='float32' else np.float64)

        n_components= self.primal.n_components#using primal attributes passed from adapter
        n_features = X.shape[1]

        if n_components>= n_features:
                raise ValueError("n_components must be < n_features;"
                                 " got %d >= %d" % (n_components, n_features))

        sess= tf.Session()#for TF  1.13
        s,u,v= sess.run(tf.linalg.svd(X, full_matrices=model_params['full_matrices'], compute_uv=model_params['compute_uv']))#for TF  1.13
        #s: singular values
        #u: normalised projection distances
        #v: decomposition/projection orthogonal axes

        v = v.T#check v is consistent with numpy's v, as tf returns adjoint v
        self.components_= v[:n_components,:]
        X_transformed = u[:,:n_components] * s[:n_components]

        self.explained_variance_= np.var(X_transformed, axis=0)
        self.singular_values_ = s[:n_components]

        #passing sigma & vh to adapter for subsequent access from adapter object itself.
        model_params={'singular_values_':self.singular_values_,'components_':self.components_}
        self.update_params(model_params)

        return X_transformed

    def transform(self, X):
        sess= tf.Session()
        res = sess.run(tf.tensordot(X, self.components_.T, axes=1))
        return res

    def inverse_transform(self, X):
        sess= tf.Session()
        res = sess.run(tf.tensordot(X, self.components_, axes=1))
        return res 

@registry.register
class LogisticRegression(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasClassifier
        self.module_name = 'sklearn'  # Rename the variable
        self.name = 'LogisticRegression'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1, ## Make key name private - '_layer'
                        'l1': 0,
                        'l2': 0,
                        'activation': 'sigmoid'},
                        'optimizer': 'adam',
                        'loss': 'binary_crossentropy'
                        }

        self.set_params(params=model_params, set_by='model_init')

# @registry.register
# class PytorchLogisticRegression(GeneralizedLinearModel):
#     '''
#     Pending -
#     1) create_model changes
#     2) adapter
#     3) optimizer changes
#     '''
#     def __init__(self):
#         self.adapter = SklearnPytorchClassifier
#         self.module_name = 'sklearn'  # Rename the variable -- primal_framework
#         # self.modeling_frontend = 'sklearn'
#         self.name = 'LogisticRegression'
#         self.version = 'V2'
#         self.modeling_backend = 'pytorch' ## proxy_framework
#         model_params = {'layer_1': {'units': 1, ## Make key name private - '_layer'
#                                     'l1': 0,
#                                     'l2': 0,
#                                     'activation': 'sigmoid'},
#                         'optimizer': 'adam',
#                         'loss': 'binary_crossentropy'
#                         } ## proxy_model_params

#         self.set_params(params=model_params, set_by='model_init')

#     def create_model(self, **kwargs):
#         import torch
#         import torch.nn as nn

#         n_in, n_h, n_out, batch_size = 10, 5, 1, 10

#         x = torch.randn(batch_size, n_in)
#         y = torch.tensor([[1.0], [0.0], [0.0], [1.0], [1.0], [1.0], [0.0], [0.0], [1.0], [1.0]])

#         model = nn.Sequential(nn.Linear(n_in, n_out), nn.Sigmoid())

#         criterion = torch.nn.MSELoss() ## try crossentropyloss instead

#         optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

#         return model

@registry.register
class LinearRegression(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasRegressor
        self.module_name = 'sklearn'
        self.name = 'LinearRegression'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1,
                                    'l1': 0,
                                    'l2': 0,
                                    'activation': 'linear'},
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        self.set_params(params=model_params, set_by='model_init')


@registry.register
class Ridge(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasRegressor
        self.module_name = 'sklearn'
        self.name = 'Ridge'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1,
                                    'l1': 0,
                                    'l2': 0.1,  # Should be configurable at tune level. Dependant on input
                                    'activation': 'linear'},
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        self.set_params(params=model_params, set_by='model_init')


@registry.register
class Lasso(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasRegressor
        self.module_name = 'sklearn'
        self.name = 'Lasso'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1,
                                    'l1': 0.1,
                                    'l2': 0,
                                    'activation': 'linear'},
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        self.set_params(params=model_params, set_by='model_init')


@registry.register
class ElasticNet(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasRegressor
        self.module_name = 'sklearn'
        self.name = 'ElasticNet'
        self.version = 'default'
        model_params = {'layer_1': {'units': 1,
                                    'l1': 0.1,
                                    'l2': 0.1,
                                    'activation': 'linear'},
                        'optimizer': 'adam',
                        'loss': 'mse'
                        }

        self.set_params(params=model_params, set_by='model_init')


@registry.register
class LinearSVC(GeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasClassifier
        self.module_name = 'sklearn'
        self.name = 'LinearSVC'
        self.version = 'default'
        model_params = {'layer_1': {
                        'l1': 0,
                        'l2': 0,
                        'activation': 'linear'},
                        'optimizer': 'adam',
                        'loss': 'categorical_hinge'
                        }

        self.set_params(params=model_params, set_by='model_init')

    def transform_data(self, X, y, y_pred):
        ## Should error handling be done at this level?
        if len(y.shape) == 1:  # Test with multiple target shapes
            y = y.reshape(-1, 1)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape([-1, 1])
        y = self.enc.transform(y)
        y = y.toarray()
        y_pred = self.enc.transform(y_pred)
        y_pred = y_pred.toarray()
        return X, y, y_pred


class KernelGeneralizedLinearModel(GeneralizedLinearModel):
    def create_model(self, **kwargs):
        model_params = _parse_params(self._model_params, return_as='nested')
        if len(self.y.shape) == 1 or self.y.shape[1] == 1:
            units = 1
        else:
            units = self.y.shape[1]
        model_params['layer_2'].update(
            {'input_dim': self.X.shape[1], 'units': units})

        model = Sequential()

        model.add(Dense(units=model_params['layer_1']['kernel_dim'],
                        trainable=False, kernel_initializer='random_normal',  # Connect with sklearn_config
                        activation=model_params['layer_1']['activation']))
        model.add(Dense(model_params['layer_2']['units'],
                        activation=model_params['layer_2']['activation']))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model


@registry.register
class SVC(KernelGeneralizedLinearModel):
    def __init__(self):
        self.adapter = SklearnKerasClassifier
        self.module_name = 'sklearn'
        self.name = 'SVC'
        self.version = 'default'
        model_params = {'layer_1': {'kernel_dim': 10,  # Make it 'units' -- Why?
                                    'activation': 'linear'
                                    },
                        'layer_2': {
                                    'activation': 'softmax'
                                    },
                        'optimizer': 'adam',
                        'loss': 'categorical_hinge'}

        self.set_params(params=model_params, set_by='model_init')

    def transform_data(self, X, y, y_pred):
        if len(y.shape) == 1:  # Test with multiple target shapes
            y = y.reshape(-1, 1)
        self.enc = OneHotEncoder(handle_unknown='ignore')
        self.enc.fit(y)
        if len(y_pred.shape) == 1:
            y_pred = y_pred.reshape([-1, 1])
        y = self.enc.transform(y)
        y = y.toarray()
        y_pred = self.enc.transform(y_pred)
        y_pred = y_pred.toarray()
        return X, y, y_pred


class CART(GeneralizedLinearModel):

    def create_model(self, **kwargs):
        model_params = _parse_params(self._model_params, return_as='nested')
        cuts_per_feature = self.cuts_per_feature
        if cuts_per_feature is None:
            feature_index, count = np.unique(
                self.primal.tree_.feature, return_counts=True)
            cuts_per_feature = np.zeros(shape=self.X.shape[1], dtype=int)

            for i, _ in enumerate(cuts_per_feature):
                for j, value in enumerate(feature_index):
                    if i == value:
                        cuts_per_feature[i] = count[j]

            cuts_per_feature = list(cuts_per_feature)

        else:
            cuts_per_feature = self.cuts_per_feature

        # if type(cuts_per_feature) not in (list, int):
        if not isinstance(cuts_per_feature, (list, int)):
            raise TypeError(
                'cuts_per_feature should be of type `list` or `int`')
        # elif type(cuts_per_feature) is int:
        elif isinstance(cuts_per_feature, int):
            if cuts_per_feature > np.ceil(self.X.shape[0]):
                cuts_per_feature = [np.ceil(self.X.shape[0]) for i in range(
                    self.X.shape[1])]
            else:
                cuts_per_feature = [cuts_per_feature for i in range(
                    self.X.shape[1])]
        else:
            if len(cuts_per_feature) != self.X.shape[1]:
                raise ValueError(
                    'The length of `cuts_per_feature` should be equal to number of features.')
            else:
                cuts_per_feature = [np.ceil(self.X.shape[0])
                                    if i > np.ceil(self.X.shape[0]) else i for i in cuts_per_feature]
        model_params['layer_3'].update({'units': self.y.shape[1]})
        visible = Input(shape=(self.X.shape[1],)) ## layer_1?
        hidden = DecisionTree(cuts_per_feature=cuts_per_feature)(visible)
        output = Dense(model_params['layer_3']['units'], activation=model_params['layer_3']['activation'])(hidden)
        model = Model(inputs=visible, outputs=output)

        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model


@registry.register
class DecisionTreeClassifier(CART):
    def __init__(self):
        self.cuts_per_feature = None
        self.adapter = SklearnKerasClassifier
        self.module_name = 'sklearn'
        self.name = 'DecisionTreeClassifier'
        self.version = 'default'
        model_params = {
            'layer_3': {'activation': 'sigmoid'},
            'optimizer': 'adam',
            'loss': 'categorical_crossentropy'
        }

        self.set_params(params=model_params, set_by='model_init')



