#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file holds utility functions used by multiple entities.
"""


def _get_module_name(model):
    return model.__class__.__module__.split('.')[0]


def _get_model_name(model):
    return model.__class__.__name__

def _parse_params(params, return_as):
    if return_as == 'nested':
        edited_params = {}
        for key, value in params.items():
            if key.split('_')[0] == 'layer': ## Add 'node' as valid prefix option
                try:
                    edited_params[key.split('.')[0]].update({key.split('.')[1]:value})
                except KeyError:
                    edited_params.update({key.split('.')[0]:{key.split('.')[1]:value}})
            else:
                edited_params.update({key:value})
    elif return_as == 'flat':
        edited_params = {}
        for key, value in params.items():
            if key.split('_')[0] == 'layer':
                for k, v in params[key].items():
                    edited_params.update({'.'.join([key, k]): v})
            else:
                edited_params.update({key:value})
    else:
        raise ValueError('Provide a valid return_as argument -- nested or flat')

    return edited_params
