#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file holds utility functions used by multiple entities.
"""


def _get_module_name(model):
    print('from functions --', model.__class__.__module__.split('.')[0])
    return model.__class__.__module__.split('.')[0]


def _get_model_name(model):
    return model.__class__.__name__
