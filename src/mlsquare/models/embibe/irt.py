from sklearn.base import BaseEstimator
import numpy as np
import pandas as pd

class fourPL(BaseEstimator):
    '''
    Input Args:
    data: a pandas dataframe (that will be converted to a numpy matrix)
    type_: a character string indicating the type of model to fit. Available options are `rasch' that assumes equal discrimination parameter among items, and `latent_trait' 
    (default) that assumes a different discrimination parameter per item.
    
    constraint: constraint must be a numpy array containing 3 columns, first col. containing tuple of items, 2nd column defines 
    type of parameter to constrain(i.e., 1 denotes the guessing parameters, 2 the easiness parameters, and 3
    the discrimination parameters); and the third column specifies the value at which the corresponding parameter should be fixed.
    example: constraint = np.array([(2, 5, 6), 2, 0.8])
    
    max_guessing: a value between 0 and 1 denoting the upper bound for the guessing parameters.
    IRT_param: Boolean type Value
    start_val:
    na_action:
    control: list type value

    `Methods`: .fit()

    Returns:
    Model signature of IRT 4PL which is to be used as primal model in dope.
    '''
    def __init__(self,data=None, type_ = "latent_trait", constraint = 'NULL', max_guessing = 1, IRT_param = True, start_val = 'NULL', na_action = 'NULL', control = list()):
        self.data=data
        self.type_ = type_
        self.constraint = constraint
        self.max_guessing= max_guessing
        self.Irt_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
    
    def fit(self, data):
        if not isinstance(self.data, (pd.core.frame.DataFrame, np.ndarray)):
            raise ValueError('Data must be either pandas Dataframe or numpy array; got (data={})'.format(type(self.data)))
        
        if type_ not in ['latent_trait', 'rasch']:
            raise ValueError('type_ must be either one of `latent_trait` for 3PL model or `rasch` for 1PL model; got (type_={})'.format(self.type_))
        
        if not isinstance(self.constraint, (np.ndarray)) and self.constraint.shape==3 and isinstance(self.constraint[0], tuple):
            raise ValueError('constraint must be a numpy array of shape 3, of which first element must be of type tuple; got (constraint of type={} and 1st element type:{})'.format(type(self.constraint),type(self.constraint[0])))

        if not isinstance(self.max_guessing, (int, float)) and 0<=self.max_guessing<=1:
            raise ValueError('max_guessing must be of type int or float and between 0 and 1; got (max_guessing type ={}, with value: {})'.format(type(self.max_guessing), self.max_guessing))

        if not isinstance(self.Irt_param, bool):
            raise ValueError('Irt_param must be either True or numpy False; got (Irt_param type={})'.format(type(self.Irt_param)))

        self.xuser= data[0]
        self.xitems= data[1]
        self._y= data[-1]

    def predict(self,data):
        return self._y


class tpm(BaseEstimator):
    '''
    Input Args:
    data: a pandas dataframe (that will be converted to a numpy matrix)
    type_: a character string indicating the type of model to fit. Available options are `rasch' that assumes equal discrimination parameter among items, and `latent_trait' 
    (default) that assumes a different discrimination parameter per item.
    
    constraint: constraint must be a numpy array containing 3 columns, first col. containing tuple of items, 2nd column defines 
    type of parameter to constrain(i.e., 1 denotes the guessing parameters, 2 the easiness parameters, and 3
    the discrimination parameters); and the third column specifies the value at which the corresponding parameter should be fixed.
    example: constraint = np.array([(2, 5, 6), 2, 0.8])
    
    max_guessing: a value between 0 and 1 denoting the upper bound for the guessing parameters.
    IRT_param: Boolean type Value
    start_val:
    na_action:
    control: list type value

    `Methods`: .fit()

    Returns:
    Model signature of IRT 3PL which is to be used as primal model in dope.
    '''
    def __init__(self,data=None, type_ = "latent_trait", constraint = 'NULL', max_guessing = 1, IRT_param = True, start_val = 'NULL', na_action = 'NULL', control = list()):
        self.data=data
        self.type_ = type_
        self.constraint = constraint
        self.max_guessing= max_guessing
        self.Irt_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
    
    def fit(self, data):
        if not isinstance(self.data, (pd.core.frame.DataFrame, np.ndarray)):
            raise ValueError('Data must be either pandas Dataframe or numpy array; got (data={})'.format(type(self.data)))
        
        if type_ not in ['latent_trait', 'rasch']:
            raise ValueError('type_ must be either one of `latent_trait` for 3PL model or `rasch` for 1PL model; got (type_={})'.format(self.type_))
        
        if not isinstance(self.constraint, (np.ndarray)) and self.constraint.shape==3 and isinstance(self.constraint[0], tuple):
            raise ValueError('constraint must be a numpy array of shape 3, of which first element must be of type tuple; got (constraint of type={} and 1st element type:{})'.format(type(self.constraint),type(self.constraint[0])))

        if not isinstance(self.max_guessing, (int, float)) and 0<=self.max_guessing<=1:
            raise ValueError('max_guessing must be of type int or float and between 0 and 1; got (max_guessing type ={}, with value: {})'.format(type(self.max_guessing), self.max_guessing))

        if not isinstance(self.Irt_param, bool):
            raise ValueError('Irt_param must be either True or numpy False; got (Irt_param type={})'.format(type(self.Irt_param)))

        self.xuser= data[0]
        self.xitems= data[1]
        self._y= data[-1]

    def predict(self,data):
        return self._y



class twoPl(BaseEstimator):
    '''
    Input Args:
    
    data: a pandas dataframe (that will be converted to a numpy matrix)
    type_: a character string indicating the type of model to fit. Available options are `rasch' that assumes equal discrimination parameter among items, and `latent_trait' 
    (default) that assumes a different discrimination parameter per item.
        
    constraint: constraint must be a numpy array containing 3 columns, first col. containing tuple of items, 2nd column defines 
    type of parameter to constrain(i.e., 1 denotes the guessing parameters, 2 the easiness parameters, and 3
    the discrimination parameters); and the third column specifies the value at which the corresponding parameter should be fixed.
    example: constraint = np.array([(2, 5, 6), 2, 0.8])

    IRT_param: Boolean type Value
    start_val:
    na_action:
    control: list type value

    `Methods`: .fit()

    Returns:
    Model signature of IRT 2PL which is to be used as primal model in dope.
    '''

    def __init__(self, data=None, constraint = 'NULL', IRT_param = True, start_val = 'NULL', na_action = 'NULL', control = list(), Hessian = True):
        self.data=data
        self.constraint = constraint
        self.IRT_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
        self.Hessian=Hessian

    def fit(self, data):
        if not isinstance(self.data, (pd.core.frame.DataFrame, np.ndarray)):
            raise ValueError('Data must be either pandas Dataframe or numpy array; got (data={})'.format(type(self.data)))
               
        if not isinstance(self.constraint, (np.ndarray)) and self.constraint.shape==3 and isinstance(self.constraint[0], tuple):
            raise ValueError('constraint must be a numpy array of shape 3, of which first element must be of type tuple; got (constraint of type={} and 1st element type:{})'.format(type(self.constraint),type(self.constraint[0])))

        if not isinstance(self.Irt_param, bool):
            raise ValueError('Irt_param must be either True or numpy False; got (Irt_param type={})'.format(type(self.Irt_param)))

        self.xuser= data.iloc[0][0]
        self.xitems= data.iloc[1][0]
        self._y= data.iloc[-1][0]

    def predict(self,data):
        return self._y

class rasch(BaseEstimator):
    '''
    Input Args:

    data: a pandas dataframe (that will be converted to a numpy matrix)
    constraint:
    IRT_param:
    start_val:
    na_action:
    control: list type value
    Hessian: Boolean type Value

    `Methods`: .fit()

    Returns:
    Model signature of IRT 1PL which is to be used as primal model in dope.
    '''
    def __init__(self, data=None, constraint = 'NULL', IRT_param = True, start_val = 'NULL', na_action = 'NULL', control = list(), Hessian = True):
        self.data=data
        self.constraint = constraint
        self.IRT_param= IRT_param
        self.start_val= start_val
        self.na_action= na_action
        self.control=control
        self.Hessian=Hessian

    def fit(self, data):
        if not isinstance(self.data, (pd.core.frame.DataFrame, np.ndarray)):
            raise ValueError('Data must be either pandas Dataframe or numpy array; got (data={})'.format(type(self.data)))
               
        if not isinstance(self.constraint, (np.ndarray)) and self.constraint.shape==3 and isinstance(self.constraint[0], tuple):
            raise ValueError('constraint must be a numpy array of shape 3, of which first element must be of type tuple; got (constraint of type={} and 1st element type:{})'.format(type(self.constraint),type(self.constraint[0])))

        if not isinstance(self.Irt_param, bool):
            raise ValueError('Irt_param must be either True or numpy False; got (Irt_param type={})'.format(type(self.Irt_param)))

        self.xuser= data[0]
        self.xitems= data[1]
        self._y= data[-1]

    def predict(self,data):
        return self._y