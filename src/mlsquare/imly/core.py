#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the base file that serves as the core of MLSquare's IMLY Module.
This exposes the function `dope`. Dope transpiles any given model to it's DNN(Deep Neural Networks) equivalent.
"""

import json
import copy
from .commons.functions import _get_model_name, _get_module_name


def dope(primal_model,abstract_model=None, wrapper=None, **kwargs): ## Rename model to primal_model?
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
        # import pkg_resources
        # resource_package = __name__
        # config_path = '/'.join(('config', 'dnn.config'))

        # get list of supported packages and algorithms
        # config = json.load(
        #     open(pkg_resources.resource_filename(resource_package, config_path)))

        # module = _get_model_module(model)
        # model_name = _get_model_class(model)

        module_name = _get_module_name(primal_model)
        model_name = _get_model_name(primal_model)

        # Check if imly support module/package used by the user.
        # if (module in config.keys() and model_name in config[module_name]):
        print("Transpiling your model to it's Deep Neural Network equivalent...")
        ## Raise as a notification(like tf)
        primal = copy.deepcopy(primal_model)

        # Get the model architecture and params + hyperparams
        # If None return primal

        # Params - Fixed parameters of the model. Not updated during runtime.
        # Fixed across all runs of the model while searching for the best model
        # Hyperparams - Optimization level parameters. Can vary during model search.
        # Static(Params/Hyperparameters) - Available without access to data and primal model.
        # Dynamic(Params/Hyperparams) - Only known once you have the data and primal model.

        # Dope is accountable for:
        # 1) Creation of model - Dealing with model_arch and params
        # 2) **Communicating with the wrapper**
        # 3) ModelMiddleware - Provides static params and hyperparams. Also, Arch for the model
        from .base import registry
        # This check should be moved to wrappers. The model_arch should get 
        # updated at that level.
        # if model_name is 'SVC':
        #     if primal.get_params()['kernel'] is 'linear':
        #         model_name = 'LinearSVC'
        #     elif primal.get_params()['kernel'] is not 'rbf':
        #         raise ValueError('{} kernel is not supported by IMLY right now'.format(primal.get_params()['kernel']))
        # model_architecture, model_params = _get_architecture(module, model_name)
        # if model_architecture and model_params:
        #     build_fn = ModelMiddleware(fn=model_architecture,
        #                                 params=model_params,
        #                                 primal=primal)
        # else:
        #     print("Unable to find a relevent dnn model architecture for the model you provided.\
        #         Hence, returning the model without transpiling.")
        #     return primal

        # Get the model wrapper class
        # If None return primal
        # from .wrappers import _get_wrapper_class
        # wrapper_class = _get_wrapper_class(module, model_name)
        print('from dope -- ', module_name, model_name)
        abstract_model, wrapper = registry[(module_name, model_name)]['default']
        # Overwrite 'default' with version if necessary

        # if wrapper_class: pass this check to BaseModel or Registry
        model = wrapper(abstract_model=abstract_model, primal=primal)
        # else:
        #     print("Unable to find a relevent wrapper function for the model you provided.\
        #         Hence, returning the model without transpiling.")
        #     return primal

        # Return the model as it is if required modules are not installed
        # if not model:
        #     print("Returning the model without transpiling since the required modules \
        #     were not installed.")
        #     return primal

        return model
        # else:
        #     print("%s from the package %s is not yet supported" %
        #           (model_name, module))
        #     print("Returing the model without transpiling")  # complete
        #     return model
        # return model
    else:
        print("Transpiling the model using %s is not yet supported. We support 'dnn' as of now" % (
            kwargs['using']))
        return model

'''
Arch refactoring TODOs
1) Fixing IMLY nomenclature - model(static/dynamic), params(static/dynamic/hyperparams)
2) Exhaustive test cases. Check pysyft examples
3) Proper error handling
4) Move core code from __init__ to it's respective files
5) y_train and x_train -- X, y
'''