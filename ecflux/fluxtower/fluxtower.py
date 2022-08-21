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
import warnings
import numpy as np
import pandas as pd

from fluxtower.utils import VARIABLES
from fluxtower import utils

class FluxTower():

    def __init__(self, filepath, meta_file=None, biomet_files=None):
        
        # _name
        # self._name = self._set_name()

        # _filepath
        self._filepath = filepath
        # _meta_file
        self._meta_file = meta_file
        # _biomet_files
        # self._biomet_files = biomet_files
        # metadata
        self.metadata = None
        # flux
        self.flux = None
        # biomet
        if biomet_files:
            self.set_biomet(biomet_files)
        else:
            self.biomet = None
            # warnings.warn("No biomet files passed. Proceeding without biomet data.")

        # data
        self.data = None

        # _var_dict
        self._var_dict = self._get_var_dict()

        # lat,lon,alt
        if self.metadata:
            self.set_coords()
        # utc_offset
            self.set_tz()
    

    def __repr__(self):
        class_name = type(self).__name__
        return '{}'.format(class_name)

    def get_metadata(self, meta_file):
        raise NotImplementedError
    
    # def import_flux(self, file, **kwargs):

    #     flux = pd.read_csv(file, **kwargs)
    #     return flux
    def import_flux(self):
        raise NotImplementedError
    
    def set_biomet(self, biomet_files):

        if isinstance(biomet_files, str):
            self.biomet = utils.import_dat(biomet_files)
        elif isinstance(biomet_files, list):
            self.biomet_list = [ utils.import_dat(file) for file in biomet_files ]
            self.biomet = self.biomet_list[0].join(self.biomet_list[1:], how='inner')

    def set_coords(self):

        self.lat = float(self.metadata.get('LAT'))
        self.lon = float(self.metadata.get('LONG'))
        self.alt = float(self.metadata.get('ELEV'))

    def set_tz(self):
        self.utc_offset = self.metadata.get('UTC_OFFSET')
    

    def _get_var_dict(self):

        try:
            var_dict = utils.get_var_dict(type(self).__name__)
        except:
            var_dict = VARIABLES
        return var_dict
    
    def _update_var_dict(self):

        # # Get variable dictionary from master dataframe
        # var_dict = utils.get_var_dict(type(self).__name__)
        # Update dictionary with actual column names
        self._var_dict.update({
            key : (self._get_var_cols(var[0]), var[1]) for key,var in self._var_dict.items() if var[0]
        })

    def _get_var_cols(self, var):

        col_list = list(self.data.columns)

        if var in col_list:
            return var
        else:
            cols = utils.get_recols(var, col_list)
            if cols:
                return cols
            else:
                return None

    def set_data(self):

        self.data = self.flux.join(
            self.biomet.resample('30T').mean(),
            # on = 'date_time',
            rsuffix = 'MET'
        )

        self._update_var_dict()

