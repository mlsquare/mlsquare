#!/usr/bin/env python
# -*- coding: utf-8 -*-


class ModelMiddleware:
    def __init__(self, fn, params, **kwargs):
        self.fn = fn
        self.params = params
        self.x_train = None

    def __call__(self, **kwargs):
        kwargs.setdefault('params', False)
        if kwargs['params']:
            self.params.update(kwargs['params'])
        try:
            # This check is temporary. This will be moved to 'AbstractModelClass' after
            # the Architectural refactoring is done.            
            kwargs.setdefault('params', self.params)
            if kwargs['params'] != self.params:
                self.params.update(kwargs['params'])
            # 'model_params' -- Change name. Creating a lot of confusion!
            model = self.fn(model_params=self.params, x_train=kwargs['x_train'],
                            y_train=kwargs['y_train'])
            return model
        except KeyError as e:
            raise KeyError('Missing argument {} while creating model. '.format(e))


def _get_architecture(module, model_name):
    """
    Given a model name, it returns the skeleton and params

    Args:
        module (str): The module from where primal model was initialised
        model_name (str): The base model class name in the module

    Returns:
        model_skeleton (func): The base method whose behaviour changes w.r.t to the input parameters
        model_params (dict): The params that are to be used with the model_skeleton to produce the same behaviour of the primal model
    """
    dispatcher = None
    if module == 'sklearn':
        from .sklearn import dispatcher

    import json

    import pkg_resources
    resource_package = __name__
    config_path = '/model_architecture.config'

    model_architecture = json.load(
        open(pkg_resources.resource_filename(resource_package, config_path)))

    # Return
    # 1. Skeleton Function
    # 2. Params to be passed to the Skeleton Function
    if dispatcher:
        return dispatcher[model_architecture['model_skeleton'][model_name]], model_architecture['model_param'][model_name]
    else:
        return None, None


def _update_architecture():
    
    pass
