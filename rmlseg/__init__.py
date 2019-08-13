# -*- coding: utf-8 -*-

"""Top-level package for PyRMLSeg."""

__author__ = """Henri DER SARKISSIAN, Nicola VIGANÒ"""
__email__ = 'N.R.Vigano@cwi.nl'


def __get_version():
    import os.path
    version_filename = os.path.join(os.path.dirname(__file__), 'VERSION')
    with open(version_filename) as version_file:
        version = version_file.read().strip()
    return version


__version__ = __get_version()

# Import all definitions from main module.
from .rmlseg import *

from . import tomo
from . import util
