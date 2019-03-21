#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Architectures
    - Contains
"""
class ModelMiddleware:
    def __init__(self, fn, params, **kwargs):
        self.fn = fn
        self.params = params
        self.x_train = None

    def __call__(self, **kwargs):
        try:
            model = self.fn(model_params=self.params, x_train=kwargs['x_train'])
            return model
        except KeyError:
            return False


def __get_architecture(module, model_name):
    dispatcher = None
    if module == 'sklearn':
        from .sklearn import dispatcher

    """
    Given a model name, it returns the a skeleton and params
    """
    import json

    try:
        # Valid
        model_architecture = json.load(
            open('./src/mlsquare/imly/architectures/model_architecture.config'))
    except:
        model_architecture = json.load(
            open('./mlsquare/imly/architectures/model_architecture.config'))


    # Return
    # 1. Skeleton Function
    # 2. Params to be passed to the Skeleton Function
    if dispatcher:
        return dispatcher[model_architecture['model_skeleton'][model_name]], model_architecture['model_param'][model_name]
    else:
        return None, None
