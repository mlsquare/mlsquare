#!/usr/bin/env python
# -*- coding: utf-8 -*-

from abc import ABCMeta, abstractmethod
from daggit.core.base.utils import get_as_list


class CreateModelObject(object):
    __metaclass__ = ABCMeta

    def __init__(self, node):
        self.node = node

    

    @property
    @abstractmethod
    def get_static_params(self):
        raise NotImplementedError('Needs to be implemented!')

    @property
    @abstractmethod
    def get_static_arch(self):
        raise NotImplementedError('Needs to be implemented!')

    @abstractmethod
    def get_dynamic_params(self)
        raise NotImplementedError('Needs to be implemented!')

    def get_dynamic_arch(self):
        raise NotImplementedError('Needs to be implemented!')

    def get_model(self):
        raise NotImplementedError('Needs to be implemented!')
            


