#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from tensorflow import set_random_seed
# from numpy.random import seed
# seed(3)
# set_random_seed(3)

from ..wrappers.base import AbstractModelClass

class generic_linear_model(AbstractModelClass):

    def __init__(self, module, model_name):
        self.module = module
        self.model_name = model_name
        self.dynamic_params = None

    
    def get_static_params(self):
        from . import _check_model_availabiltiy

        model_architecture, static_params = _check_model_availabiltiy(self.module, self.model_name)
        # self.static_model = ModelMiddleware(fn=model_architecture,
        #                                     params=static_params)
        return static_params

    def get_static_arch(self): # Purpose?
        pass

    def set_dynamic_params(self, params):
        self.dynamic_params = params

    def get_dynamic_params(self):
        return self.dynamic_params

    def get_dynamic_arch(self): # Purpose?
        pass

    def get_model(self):
        try:
            params = self.get_static_params()
            if self.dynamic_params is not None:
                params.update(self.dynamic_params)
            
            from keras.models import Sequential
            from keras.layers.core import Dense

            model_params = kwargs['model_params']
            model = Sequential()
            model.add(Dense(model_params['units'],
                            input_dim=kwargs['x_train'].shape[1],
                            activation=model_params['activation']))
            model.compile(optimizer=model_params['optimizer'],
                          loss=model_params['losses'],
                          metrics=['accuracy'])

            return model
        except ImportError:
            print ("keras is required to transpile the model")
            return False

        



def generic_linear_model(**kwargs):
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense

        model_params = kwargs['model_params']
        model = Sequential()
        model.add(Dense(model_params['units'],
                        input_dim=kwargs['x_train'].shape[1],
                        activation=model_params['activation']))
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


dispatcher = {
    'glm': generic_linear_model,
    'lda': linear_discriminant_analysis
}
