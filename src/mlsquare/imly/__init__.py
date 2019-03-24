#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the base file that serves as the core of MLSquare's IMLY Module

This exposes the function `dope` which transpiles any given model to its implementation using Deep Neural Networks.
"""

import json
import copy
from .commons.functions import _get_model_class, _get_model_module


def dope(model, **kwargs):
    """Augments a given model using DNN with optimization

    Args:
        model (class): The primal model passed by the user that needs to be transpiled.
        using (str): Choose the algo to transpile the model

            1. None: returns the model as it is

            2. dnn (default): converts the model to run using deep neural network

        best (bool): Whether to optmize the model or not
        **kwargs (dict): Dictionary of parameters mapped to their keras params.

    Returns:
        model (class): The transpiled model.
    """

    # Set the default values for the arguments
    kwargs.setdefault('using', 'dnn')
    kwargs.setdefault('best', True)

    if (kwargs['using'] == None):
        return model

    elif (kwargs['using'] == 'dnn'):
        # get list of supported packages and algorithms
        try:
            # Valid
            config = json.load(open('./src/mlsquare/imly/config/dnn.config'))
        except:
            config = json.load(open('./mlsquare/imly/config/dnn.config'))

        module = _get_model_module(model)
        model_name = _get_model_class(model)

        # Check for the imly support for the module
        if (module in config.keys() and model_name in config[module]):
            print("Transpiling the model to use Deep Neural Networks")
            primal = copy.deepcopy(model)

            # Get the model architecture and params
            # If None return primal
            from .architectures import ModelMiddleware, _get_architecture
            model_architecture, model_params = _get_architecture(
                module, model_name)
            if model_architecture and model_params:
                build_fn = ModelMiddleware(fn=model_architecture,
                                           params=model_params,
                                           primal=primal)
            else:
                return primal

            # Get the model wrapper class
            # If None return primal
            from .wrappers import _get_wrapper_class
            wrapper_class = _get_wrapper_class(module, model_name)
            if wrapper_class:
                model = wrapper_class(build_fn=build_fn, params=None,
                                      primal=primal,
                                      best=kwargs['best'])
            else:
                return primal

            # Return the model as it is if required modules are not installed
            if not model:
                return primal

            return model
        else:
            print("%s from the package %s is not yet supported" %
                  (model_name, module))
            print("Returing the  without transpiling")  # complete
            return model
        return model
    else:
        print("Transpiling the model using %s is not yet supported. We support 'dnn' as of now" % (
            kwargs['using']))
        return model
