#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
This is the base file that serves as the core of MLSquare's IMLY Module.
This exposes the function `dope`. Dope transpiles any given model to it's DNN(Deep Neural Networks) equivalent.
"""

import json
import copy
import sys
from .utils.functions import _get_model_name, _get_module_name
from .base import registry

def dope(primal_model, abstract_model=None, adapt=None, **kwargs): ## Rename model to primal_model?
    """Transpiles a given model to it's DNN equivalent.

    Args:
        model (class): The primal model passed by the user that needs to be transpiled.
        using (str): Choice of type of "model transpilation" you want your model to undergo.
        Currently accepts None and 'dnn' as values.

            1. None: Returns the model as it is.

            2. dnn (default): Converts the model to it's DNN equivalent.

        **kwargs (dict): Dictionary of parameters mapped to their keras params.

    Returns:
        model (class): The transpiled model.
    """

    # Set the default values for the arguments
    kwargs.setdefault('using', 'dnn')
    kwargs.setdefault('version', 'default')
    model_version = kwargs['version']

    if (kwargs['using'] == None):
        print('Returning model without converting to its proxy dnn version. ', file=sys.stderr)
        return primal_model

    elif (kwargs['using'] == 'dnn'):

        module_name = _get_module_name(primal_model)
        model_name = _get_model_name(primal_model)

        primal = copy.deepcopy(primal_model) # Needed as a copy?
        if abstract_model == None and adapt == None:
            try:
                abstract_model, adapt = registry[(module_name, model_name)][model_version]
            except KeyError:
                # raise TypeError('Model type `%s` is not supported by mlsquare yet.' % (type(primal_model)))
                raise TypeError('Unsupported model or version. Please check your model type and version' % (type(primal_model)))

        print("Transpiling your model to it's Deep Neural Network equivalent...", file=sys.stderr)
        model = adapt(abstract_model=abstract_model, primal=primal) # Change to adapter?

        return model
    else:
        print("Transpiling models using `%s` is not yet supported. We support 'dnn' as of now." % (
            kwargs['using']), file=sys.stderr)
        return primal_model

'''
1) Error handling at component level v/s dope/higher level -- back to the caller
'''