#!/usr/bin/env python
# -*- coding: utf-8 -*-

def _get_wrapper_class(module, model_name):
    wrappers = None
    if module == 'sklearn':
        from .sklearn import wrappers

    if wrappers:
        return wrappers[model_name]
    else:
        return None