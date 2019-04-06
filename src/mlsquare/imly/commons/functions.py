#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
    This file holds utility functions used by multiple entities.
"""


def _get_model_module(model):
    return model.__class__.__module__.split('.')[0]


def _get_model_class(model):
    return model.__class__.__name__
