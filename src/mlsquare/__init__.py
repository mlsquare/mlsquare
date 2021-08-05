# -*- coding: utf-8 -*-
from pkg_resources import get_distribution, DistributionNotFound

try:
    # Change here if project is renamed and does not equal the package name
    dist_name = __name__
    __version__ = get_distribution(dist_name).version
except DistributionNotFound:
    __version__ = 'unknown'
finally:
    del get_distribution, DistributionNotFound

# __version__ = 'v0.1.1'
__author__ = "MLSquare"
__copyright__ = "MLSquare"
__license__ = "mit"

from .core import dope
from .base import registry
from .architectures import sklearn, irt
