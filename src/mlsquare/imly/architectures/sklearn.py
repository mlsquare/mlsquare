#!/usr/bin/env python
# -*- coding: utf-8 -*-


# from tensorflow import set_random_seed
# from numpy.random import seed
# seed(3)
# set_random_seed(3)

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


def linear_discriminant_analysis(**kwargs): # Refactor!
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

def decision_tree_classifier(**kwargs):
    try:
        from keras.models import Model
        from keras.layers import Input, Dense
        from ..commons.custom_layers import DecisionTree

        model_params = kwargs['model_params']

        default_cutpoints = [1 for i in range(kwargs['x_train'].shape[1])]

        kwargs.setdefault('cuts_per_feature', default_cutpoints)
        visible = Input(shape=(kwargs['x_train'].shape[1],))
        hidden = DecisionTree(cuts_per_feature=kwargs['cuts_per_feature'])(visible)
        output = Dense(model_params['units'], activation=model_params['activation'])(hidden)
        model = Model(inputs=visible, outputs=output)
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['losses'],
                      metrics=['accuracy'])

        return model
    except ImportError:
        print("keras is required to transpile the model")
        return False



dispatcher = {
    'glm': generic_linear_model,
    'lda': linear_discriminant_analysis,
    'dt_classifier': decision_tree_classifier
}
