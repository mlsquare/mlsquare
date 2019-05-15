#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from tensorflow import set_random_seed
# from numpy.random import seed
# seed(3)
# set_random_seed(3)


def glm(**kwargs): # Change to glm
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.regularizers import l2

        ## Temporary hack. Needs to be fixed during architectural refactoring.
        kwargs.setdefault('y_train', None)
        if len(kwargs['y_train'].shape) == 1: # Can be delegated to wrappers
            units = 1
        else:
            units = kwargs['y_train'].shape[1]

        model_params = kwargs['model_params']
        model = Sequential()
        model.add(Dense(units,
                        input_dim=kwargs['x_train'].shape[1],
                        activation=model_params['activation'],
                        kernel_regularizer=l2(model_params['kernel_regularizer_value'])))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print ("keras is required to transpile the model")
        return False


def linear_discriminant_analysis(**kwargs):
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.regularizers import l2
        from ..commons.losses import mse_in_theano

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

def kernel_glm(**kwargs): # Update in config
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense

        ## Temporary hack. Needs to be fixed during architectural refactoring.
        kwargs.setdefault('y_train', None)

        model_params = kwargs['model_params']
        model = Sequential()
        model.add(Dense(kwargs['y_train'].shape[1],
                        input_dim=kwargs['x_train'].shape[1],
                        activation=model_params['activation']))
        model.add(Dense(model_params['kernel_dim'], # kernel dimensions - hyperparams
                        trainable=False, kernel_initializer='random_normal', # Check for random_normal - for rbf
                        activation=model_params['activation']))
        model.add(Dense(kwargs['y_train'].shape[1],
                        activation='softmax'))
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print ("keras is required to transpile the model")
        return False


dispatcher = {
    'glm': glm,
    'lda': linear_discriminant_analysis,
    'rbf': kernel_glm
}