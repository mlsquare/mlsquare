#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the base file that serves as the core of MLSquare's IMLY Module.
This exposes the function `dope`. Dope transpiles any given model to it's DNN(Deep Neural Networks) equivalent.
"""

import json
import copy
from .utils.functions import _get_model_name, _get_module_name
from .base import registry


def dope(primal_model,abstract_model=None, adapter=None, **kwargs): ## Rename model to primal_model?
    print(primal_model.__module__)
    """Transpiles a given model to it's DNN equivalent.

    Args:
        model (class): The primal model passed by the user that needs to be transpiled.
        using (str): Choice of type of "model transpilation" you want your model to undergo.
        Currently accepts None and 'dnn' as values.

            1. None: Returns the model as it is.

            2. dnn (default): Converts the model to it's DNN equivalent.

        best (bool): Whether to optmize the model or not.
        **kwargs (dict): Dictionary of parameters mapped to their keras params.

    Returns:
        model (class): The transpiled model.
    """

    # Set the default values for the arguments
    kwargs.setdefault('using', 'dnn')
    kwargs.setdefault('best', True) # Remove. Optimization should happen by default.

    if (kwargs['using'] == None):
        ## Notify the user!
        return primal_model

    elif (kwargs['using'] == 'dnn'):

        module_name = _get_module_name(primal_model)
        model_name = _get_model_name(primal_model)

        # Check if imly support module/package used by the user.
        # if (module in config.keys() and model_name in config[module_name]):
        print("Transpiling your model to it's Deep Neural Network equivalent...")
        ## Raise as a notification(like tf)
        primal = copy.deepcopy(primal_model)
        print('from dope -- ', module_name, model_name)

        abstract_model, adapt = registry[(module_name, model_name)]['default']
        # Overwrite 'default' with version if necessary

        # if wrapper_class: pass this check to BaseModel or Registry
        model = adapt(abstract_model=abstract_model, primal=primal) ## wrapper - change name
        # model = adapt(abstract_model, primal) 

        return model
    else:
        print("Transpiling the model using %s is not yet supported. We support 'dnn' as of now" % (
            kwargs['using']))
        return model

'''
Arch refactoring TODOs
1) Fixing IMLY nomenclature - model(static/dynamic), params(static/dynamic/hyperparams) [X]
2) Exhaustive test cases. Check pysyft examples
3) Proper error handling
4) Move core code from __init__ to it's respective files [X]
5) y_train and x_train -- X, y
5) Alternate names for wrapper - 
    + module_extender
    + enhancer
    + converter(converting standard ml to dnn)
    + *adapter* -- adapt(verb)
'''