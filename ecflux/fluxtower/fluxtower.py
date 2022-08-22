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
        if self._meta_file:
            self.set_metadata()
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
        self.var_dict = self._get_var_dict()

        # # lat,lon,alt
        # if self.metadata:
        #     self.set_coords()
        # # utc_offset
        #     self.set_tz()
    

    def __repr__(self):
        class_name = type(self).__name__
        return '{}'.format(class_name)

    def set_metadata(self):

        # Read in metadata from _meta_file
        self.metadata = pd.read_csv(
            self._meta_file, header=None, index_col=0
        )[1].to_dict()
        # Convert numbers to numeric
        self.metadata.update(
            (k, pd.to_numeric(v, errors='ignore')) for k,v in self.metadata.items()
        )

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
        self.var_dict.update({
            key : (self._get_var_cols(var[0]), var[1]) for key,var in self.var_dict.items() if var[0]
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

    def calc_avg(self, var):

        cols = self.var_dict.get(var)[0]
        avg = self.data[cols].mean(axis = 1)

        return avg
    
    def get_highest(self):
        raise NotImplementedError
    
    def clean_data(self):
        """
        Clean self.data. The following actions are performed: 
            1. Average replicated measurements (e.g. G, T_c)
            2. Convert units if necessary (e.g. T, p_a)
            3. Extract the highest measurement + average if necessary (only applies
               to AmeriFlux data; e.g. u, SW_IN, T_a)
            4. Rename columns

        # TODO: Make this better. Would like get_highest to go in AmeriFluxTower class.
        # TODO: Refactor to deal with profile measurements.
        # TODO: Refactor to better handle unit conversions (current way is very bad programming).
        """
        for var,unit in list(VARIABLES.items())[1:]:

            cols = self.var_dict.get(var)[0]
            units = self.var_dict.get(var)[1]
            
            # SINGLE COLUMN
            if isinstance(cols, str):
                # if var == cols:
                #     pass
                # If units are not the same, convert units + add new column
                if unit != units:
                    self.data[var] = utils.convert_units(self.data[cols], units)
                # Otherwise, rename the existing column
                else:
                    self.data.rename(columns={cols:var}, inplace=True)
            # MULTIPLE COLUMNS
            elif isinstance(cols, list):
                if var == 'G':
                    self.data[var] = self.calc_avg(var)
                elif var == 'T_c':
                    self.data[var] = utils.convert_units(self.calc_avg(var), units)
                elif var == 'T_a':
                    self.data[var] = utils.convert_units(self.get_highest(var), units)
                else:
                    self.data[var] = self.get_highest(var)

