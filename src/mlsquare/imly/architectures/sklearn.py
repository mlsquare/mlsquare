# #!/usr/bin/env python
# # -*- coding: utf-8 -*-


# # from tensorflow import set_random_seed
# # from numpy.random import seed
# # seed(3)
# # set_random_seed(3)


# def glm(**kwargs): # Change to glm
#     try:
#         from keras.models import Sequential
#         from keras.layers.core import Dense
#         from keras.regularizers import l1_l2

#         ## Temporary hack. Needs to be fixed during architectural refactoring.
#         kwargs.setdefault('y_train', None)
#         if len(kwargs['y_train'].shape) == 1: # Can be delegated to wrappers
#             units = 1
#         else:
#             units = kwargs['y_train'].shape[1]

#         model_params = kwargs['model_params']
#         model = Sequential()
#         model.add(Dense(units,
#                         input_dim=kwargs['x_train'].shape[1],
#                         activation=model_params['activation'],
#                         kernel_regularizer=l1_l2(l1=model_params['l1'], l2=model_params['l2']))) # Update as l1() for lasso
#         model.compile(optimizer=model_params['optimizer'],
#                       loss=model_params['losses'],
#                       metrics=['accuracy'])

#         return model
#     except ImportError:
#         print ("keras is required to transpile the model")
#         return False


# def linear_discriminant_analysis(**kwargs): # Refactor!
#     try:
#         from keras.models import Sequential
#         from keras.layers.core import Dense
#         from keras.regularizers import l2
#         from ..commons.losses import mse_in_theano

#         model = Sequential()
#         model.add(Dense(
#             kwargs['params']['units'],
#             input_dim=kwargs['x_train'].shape[1],
#             activation=kwargs['params']['activation'][0],
#             kernel_regularizer=l2(1e-5)
#         ))
#         model.compile(
#             optimizer=kwargs['params']['optimizer'],
#             loss=mse_in_theano,
#             metrics=['accuracy']
#         )

#         return model
#     except ImportError:
#         print("keras is required to transpile the model")
#         return False

# def cart(**kwargs):
#     try:
#         from keras.models import Model
#         from keras.layers import Input, Dense
#         import numpy as np
#         from ..commons.layers import DecisionTree

#         ## This validation is not moved to the DT Layer since x_train.shape[0] is not
#         ## available at that level.

#         cuts_per_feature = kwargs['model_params']['cuts_per_feature'] 
#         # Update to kwargs['model_params']['cuts_per_feature']
#         if type(cuts_per_feature) not in (list, int):
#             raise TypeError('cuts_per_feature should be of type `list` or `int`')
#         elif type(cuts_per_feature) is int:
#             if cuts_per_feature > np.ceil(kwargs['x_train'].shape[0]):
#                 cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0]) for i in range(kwargs['x_train'].shape[1])]
#             else:
#                 cuts_per_feature = [cuts_per_feature for i in range(kwargs['x_train'].shape[1])]
#         else:
#             if len(cuts_per_feature) != kwargs['x_train'].shape[1]:
#                 print("From arch -- ",cuts_per_feature)
#                 raise ValueError('The length of `cuts_per_feature` should be equal to number of features.')
#             else:
#                 cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0]) 
#                                     if i > np.ceil(kwargs['x_train'].shape[0]) else i for i in cuts_per_feature]


#         model_params = kwargs['model_params']
#         kwargs.setdefault('units', kwargs['model_params']['units'])
#         visible = Input(shape=(kwargs['x_train'].shape[1],))
#         # hidden = DecisionTree(cuts_per_feature=kwargs['cuts_per_feature'])(visible)
#         hidden = DecisionTree(cuts_per_feature=cuts_per_feature)(visible)
#         output = Dense(kwargs['units'], activation=model_params['activation'])(hidden)
#         model = Model(inputs=visible, outputs=output)

#         model.compile(optimizer=model_params['optimizer'],
#                 loss=model_params['losses'],
#                 metrics=['accuracy'])

#         return model
#     except ImportError:
#         print("keras is required to transpile the model")
#         return False # Raise error instead of returning False. False doesn't help much while debugging.


# def kernel_glm(**kwargs): # Update in config
#     try:
#         from keras.models import Sequential
#         from keras.layers.core import Dense

#         ## Temporary hack. Needs to be fixed during architectural refactoring.
#         kwargs.setdefault('y_train', None)

#         model_params = kwargs['model_params']
#         model = Sequential()
#         model.add(Dense(kwargs['y_train'].shape[1],
#                         input_dim=kwargs['x_train'].shape[1],
#                         activation=model_params['activation']))
#         model.add(Dense(model_params['kernel_dim'], # kernel dimensions - hyperparams
#                         trainable=False, kernel_initializer='random_normal', # Check for random_normal - for rbf
#                         activation=model_params['activation']))
#         model.add(Dense(kwargs['y_train'].shape[1],
#                         activation='softmax'))
#         model.compile(optimizer=model_params['optimizer'],
#                       loss=model_params['losses'],
#                       metrics=['accuracy'])

#         return model
#     except ImportError:
#         print ("keras is required to transpile the model")
#         return False


# dispatcher = {
#     'glm': glm,
#     'lda': linear_discriminant_analysis,
#     'rbf': kernel_glm,
#     'cart': cart
# }


# # # # # # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # ## # # #



#!/usr/bin/env python
# -*- coding: utf-8 -*-

from ..base import registry, BaseModel
from keras.models import Sequential
from keras.layers.core import Dense
from keras.regularizers import l1_l2
from ..wrappers.sklearn import SklearnKerasClassifier
from ..base import registry


class GenericLinearModel(BaseModel): ## Rename glm

    def create_model(self):

        model_params = self._model_params ## Why make it private? Alternate name?
        model = Sequential()
        model.add(Dense(units=model_params['units'],
                        input_dim=model_params['input_dim'],
                        activation=model_params['activation'])) # Update as l1() for lasso
        model.compile(optimizer=model_params['optimizer'],
                      loss=model_params['loss'],
                      metrics=['accuracy'])

        return model

    def set_params(self, params):
        print(params)
        self._model_params = params


    def get_params(self):
        return self._model_params

    def wrapper(self):
        return self._wrapper

@registry.register
class LogisticRegression(GenericLinearModel):
    def __init__(self):
        self.wrapper = SklearnKerasClassifier
        # wrapper = self.wrapper
        self.module_name = 'sklearn' # Can be accessed directly from primal model

    def create_model(self):
        model_params = { 'units': 1,
                        'input_dim': 10,
                        'activation': 'sigmoid',
                        'optimizer': 'adam',
                        'loss': 'binary_crossentropy'
                        }

        self.set_params(model_params)
        return super().create_model()

    # Add an update params method(to update units and input_dims)




def glm(**kwargs): # Change to glm
    try:
        from keras.models import Sequential
        from keras.layers.core import Dense
        from keras.regularizers import l1_l2

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
                        kernel_regularizer=l1_l2(l1=model_params['l1'], l2=model_params['l2']))) # Update as l1() for lasso
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

def cart(**kwargs):
    try:
        from keras.models import Model
        from keras.layers import Input, Dense
        import numpy as np
        from ..commons.layers import DecisionTree

        ## This validation is not moved to the DT Layer since x_train.shape[0] is not
        ## available at that level.

        cuts_per_feature = kwargs['model_params']['cuts_per_feature'] 
        # Update to kwargs['model_params']['cuts_per_feature']
        if type(cuts_per_feature) not in (list, int):
            raise TypeError('cuts_per_feature should be of type `list` or `int`')
        elif type(cuts_per_feature) is int:
            if cuts_per_feature > np.ceil(kwargs['x_train'].shape[0]):
                cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0]) for i in range(kwargs['x_train'].shape[1])]
            else:
                cuts_per_feature = [cuts_per_feature for i in range(kwargs['x_train'].shape[1])]
        else:
            if len(cuts_per_feature) != kwargs['x_train'].shape[1]:
                print("From arch -- ",cuts_per_feature)
                raise ValueError('The length of `cuts_per_feature` should be equal to number of features.')
            else:
                cuts_per_feature = [np.ceil(kwargs['x_train'].shape[0]) 
                                    if i > np.ceil(kwargs['x_train'].shape[0]) else i for i in cuts_per_feature]


        model_params = kwargs['model_params']
        kwargs.setdefault('units', kwargs['model_params']['units'])
        visible = Input(shape=(kwargs['x_train'].shape[1],))
        # hidden = DecisionTree(cuts_per_feature=kwargs['cuts_per_feature'])(visible)
        hidden = DecisionTree(cuts_per_feature=cuts_per_feature)(visible)
        output = Dense(kwargs['units'], activation=model_params['activation'])(hidden)
        model = Model(inputs=visible, outputs=output)

        model.compile(optimizer=model_params['optimizer'],
                loss=model_params['losses'],
                metrics=['accuracy'])

        return model
    except ImportError:
        print("keras is required to transpile the model")
        return False # Raise error instead of returning False. False doesn't help much while debugging.


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
