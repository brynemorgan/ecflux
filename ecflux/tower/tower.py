#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Wed 17 Aug 22 13:32:19'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           tower.py
Compatibility:  Python 3.10.2
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2022


"""


# IMPORTS 
import os
import numpy as np
import pandas as pd


class FluxTower():

    def __init__(self, filepath=None, meta_file=None):

        self._filepath = filepath

        self._meta_file = meta_file

        self.metadata = None

        self.fluxdata = None

        self.met = None

    
    def get_metadata(self, meta_file):
        raise NotImplementedError
    
    def import_flux(self):
        raise NotImplementedError
    
    def import_biomet(self):
        raise NotImplementedError
    
    def get_met(self):
        raise NotImplementedError



