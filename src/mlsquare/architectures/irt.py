import numpy as np
import keras
from keras import backend as K
from ..base import registry, BaseModel
from ..adapters.sklearn import IrtKerasRegressor
from keras.layers import Dense, Input
from keras.regularizers import l1_l2
from keras.models import Model
from ..utils.functions import _parse_params
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
                kernel_initializer= model_params['ability_params']['kernel_init'],
                kernel_regularizer=l1_l2(l1=model_params['regularizers']['l1'], l2= model_params['regularizers']['l2']),
                name='latent_trait')(user_input_layer)

        #2. kernel init set to RandomNorm(0,1)
        #b_j
        difficulty_level = Dense(model_params['ability_params']['units'], use_bias=False,
            kernel_initializer= model_params['diff_params']['kernel_init'],
            name='difficulty_level')(quest_input_layer)#b_j

        #3. kernel init set to RandomNorm(1,1)
        # Descrimination- also lamda_j
        discrimination_param = Dense(model_params['ability_params']['units'], use_bias=False,
            kernel_initializer= model_params['disc_params']['kernel_init'],
            trainable=model_params['disc_params']['train'],
            activation= model_params['disc_params']['act'],
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
            kernel_initializer= model_params['guess_params']['kernel_init'],
            trainable=model_params['guess_params']['train'],
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
