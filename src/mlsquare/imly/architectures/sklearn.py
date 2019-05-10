#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from tensorflow import set_random_seed
# from numpy.random import seed
# seed(3)
# set_random_seed(3)

class generic_linear_model(CreateModelObject):
    
    def get_static_params(self):
        pass

    def get_static_arch(self):
        pass

    def get_dynamic_params(self):
        pass

    def get_dynamic_arch(self):
        pass

    def get_model(self):
        try:
            params = self.get_static_params()
            params = self.get_static_params()
            
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
