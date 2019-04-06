#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the base file that serves as the core of MLSquare's IMLY Module.
This exposes the function `dope`. Dope transpiles any given model to it's DNN(Deep Neural Networks) equivalent.
"""

import json
import copy
from .commons.functions import _get_model_class, _get_model_module


def dope(model, **kwargs):
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
    kwargs.setdefault('best', True)

    if (kwargs['using'] == None):
        return model

    elif (kwargs['using'] == 'dnn'):
        import pkg_resources
        resource_package = __name__
        config_path = '/'.join(('config', 'dnn.config'))

        # get list of supported packages and algorithms
        config = json.load(
            open(pkg_resources.resource_filename(resource_package, config_path)))

        module = _get_model_module(model)
        model_name = _get_model_class(model)

        # Check if imly support module/package used by the user.
        if (module in config.keys() and model_name in config[module]):
            print("Transpiling your model to it's Deep Neural Network equivalent")
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
                print("Unable to find a relevent dnn model architecture for the model you provided.\
                    Hence, returning the model without transpiling.")
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
                print("Unable to find a relevent wrapper function for the model you provided.\
                    Hence, returning the model without transpiling.")
                return primal

            # Return the model as it is if required modules are not installed
            if not model:
                print("Returning the model without transpiling since the required modules \
                were not installed.")
                return primal

            return model
        else:
            print("%s from the package %s is not yet supported" %
                  (model_name, module))
            print("Returing the model without transpiling")  # complete
            return model
        return model
    else:
        print("Transpiling the model using %s is not yet supported. We support 'dnn' as of now" % (
            kwargs['using']))
        return model
