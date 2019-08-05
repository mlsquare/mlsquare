#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
import sys
from .utils.functions import _get_model_name, _get_module_name
from .base import registry

def dope(primal_model, proxy_model=None, adapter=None, **kwargs): ## Rename model to primal_model?
    """Transpiles a given model to it's DNN equivalent.

    Parameters
    ----------
    primal_model : Primal model instance
        The model that needs to be transpiled.

    proxy_model : Proxy model class, optional
        The dnn equivalent of the given primal model.
        This is used only if you intend to provide the proxy model
        externally.

    adapter : Adapter class, optional
        Corresponding adapter for the proxy model.

    version : str, optional
        Choice of version of proxy model. Default is 'default'.

    Raises
    ------
    TypeError
        If unsupported primal model or version type is passed.

    Returns
    -------
    model : Mapped proxy model instance
        The final transpiled model.
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
        if proxy_model == None and adapter == None:
            try:
                proxy_model, adapter = registry[(module_name, model_name)][model_version]
            except KeyError:
                # raise TypeError('Model type `%s` is not supported by mlsquare yet.' % (type(primal_model)))
                raise TypeError('Unsupported model or version. Please check your model type and version' % (type(primal_model)))
        elif proxy_model != None and adapter == None:
            raise ValueError('Please pass a valid adapter for your primal model')
        elif proxy_model == None and adapter != None:
            raise ValueError('Please pass a valid primal model with your adapter')

        print("Transpiling your model to it's Deep Neural Network equivalent...", file=sys.stderr)
        model = adapter(proxy_model=proxy_model, primal_model=primal)

        return model
    else:
        print("Transpiling models using `%s` is not yet supported. We support 'dnn' as of now." % (
            kwargs['using']), file=sys.stderr)
        return primal_model

# TODO
# Update proxy and primal in adapters and optim