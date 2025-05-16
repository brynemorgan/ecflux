#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Wed 17 Aug 22 16:39:17'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           amf_tower.py
Compatibility:  Python 3.7.0
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

from fluxtower import FluxTower,utils
from fluxtower.utils import cols_to_dict


# VARIABLES

AMF_META_VARS = {
    'AMF_ID' : 'AMF_ID',
    'SITE_NAME' : 'SITE_NAME',
    'DATE_START' : 'FLUX_MEASUREMENTS_DATE_START',
    'DATE_END' : 'FLUX_MEASUREMENTS_DATE_END',
    'MAT' : 'MAT',
    'MAP' : 'MAP',
    'CLIMATE_KOEPPEN' : 'CLIMATE_KOEPPEN',
    'IGBP' : 'IGBP',
    'LAT' : 'LOCATION_LAT',
    'LONG' : 'LOCATION_LONG',
    'ELEV' : 'LOCATION_ELEV',
    'UTC_OFFSET' : 'UTC_OFFSET',
    # 'VEG_HEIGHT' : 'HEIGHTC',
}

AMF_SUPP_META = {
    'CR-SoC' : {
        'SITE_ID' : 'SOLC',
        'VEG_HEIGHT' : None,
        'TOWER_HEIGHT' : None
    },
    'US-Wkg' : {
        'SITE_ID' : 'WALN',
        'VEG_HEIGHT' : 0.5,
        'TOWER_HEIGHT' : 6.4
    },
    'US-xBR' : {
        'SITE_ID' : 'BART',
        'VEG_HEIGHT' : 23.0,
        'TOWER_HEIGHT' : 35.68
    },
    'US-xGR' : {
        'SITE_ID' : 'GRSM',
        'VEG_HEIGHT' : 30.0,
        'TOWER_HEIGHT' : 45.0
    },
    'US-xHA' : {
        'SITE_ID' : 'HARV',
        'VEG_HEIGHT' : 26.0,
        'TOWER_HEIGHT' : 26.0
    },
    'US-xRN' : {
        'SITE_ID' : 'ORNL',
        'VEG_HEIGHT' : 28.0,
        'TOWER_HEIGHT' : 28.0
    },
    'US-xSE' : {
        'SITE_ID' : 'SERC',
        'VEG_HEIGHT' : 38.0,
        'TOWER_HEIGHT' : 60.0
    },
    'US-xSJ' : {
        'SITE_ID' : 'SJER',
        'VEG_HEIGHT' : 21.0,
        'TOWER_HEIGHT' : 39.0
    },
}

# Numeric columns (incomplete list)
num_cols = ['MAT', 'MAP', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV', 'UTC_OFFSET', 'HEIGHTC']

# def cols_to_dict(df, key_col='VARIABLE', val_col='DATAVALUE'):

#     return pd.Series(df[val_col].values,index=df[key_col]).to_dict()

# Raw AMF BIF data
AMF_BADM_DF = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        # 'AMF_AA-Net_BIF_CCBY4_20220811.csv'
        'AMF_AA-Net_BIF_CCBY4_20250430.csv'
    )
)# AMF BADM data for all sites (dict of dicts)
AMF_SITE_BADM = AMF_BADM_DF.groupby('SITE_ID').apply(
    lambda group : cols_to_dict(group),
    include_groups=False
).to_dict()


AMF_HEIGHT_DF = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        # 'BASE_MeasurementHeight_20220811.csv'
        'BASE_MeasurementHeight_20250430.csv'
    )
)

AMF_SITE_HEIGHT = AMF_HEIGHT_DF.groupby('Site_ID').apply(
    lambda group : cols_to_dict(group, key_col='Variable', val_col='Height'),
    include_groups=False
).to_dict()




class AmeriFluxTower(FluxTower):

    cls_dict = utils.get_var_dict('AmeriFluxTower')

    def __init__(self, filepath, amf_id=None):

        super().__init__(filepath)

        # amf_id
        if amf_id:
            self.id = amf_id
        else:
            self.id = os.path.basename(self._filepath)[4:10]

        # badm
        self.set_badm()
                
        # metadata
        self.set_metadata()

        # heights
        self.set_heights()
        
        # _base_file
        self._base_file = self.get_flux_file()

        # flux
        self.flux = self.import_flux()

        # data
        self.set_data()
        self.clean_data()


    def set_metadata(self):

        self.metadata = {k : self.badm.get(v, None) for k,v in AMF_META_VARS.items()}
        if self.id in AMF_SUPP_META.keys():
            self.metadata.update(AMF_SUPP_META.get(self.id))
        
        self._set_coords()
        self._set_tz()

    def set_badm(self):

        # self.badm = AMF_BADM_DB[AMF_BADM_DB.SITE_ID == self.id].to_dict(orient='records')
        self.badm = {'AMF_ID' : self.id} | AMF_SITE_BADM.get(self.id)
        self.badm.update((k, pd.to_numeric(v,errors='ignore')) for k, v in self.badm.items())
    
    def set_heights(self):
            
        self.heights = AMF_SITE_HEIGHT.get(self.id)
    
    def _get_all_badm(self):
        return AMF_BADM_DF[AMF_BADM_DF.SITE_ID == self.id]
    
    def _get_all_heights(self):
        return AMF_HEIGHT_DF[AMF_HEIGHT_DF.Site_ID == self.id]

    def get_flux_file(self):

        file = [file for file in os.listdir(self._filepath) if 'BASE' in file][0]

        return os.path.join(self._filepath,file)


    def import_flux(self, skiprows=2, na_values=-9999., **kwargs):

        '''
        NOTE: Would like to parse dates on import (see below), but this is ~9x slower than
        converting dates afterwards...
        flux = pd.read_csv(base_file, skiprows=2, parse_dates=[0,1], infer_datetime_format=True, na_values=-9999.)

        '''

        flux = pd.read_csv(self._base_file, skiprows=skiprows, na_values=na_values, **kwargs)

        flux.TIMESTAMP_START = pd.to_datetime(flux.TIMESTAMP_START, format='%Y%m%d%H%M')
        flux.TIMESTAMP_END = pd.to_datetime(flux.TIMESTAMP_END, format='%Y%m%d%H%M')

        dt_named = self._set_col_tz(flux.TIMESTAMP_END)

        flux.set_index(dt_named, inplace=True)

        return flux


    def _get_var_cols(self, var):

        col_list = list(self.data.columns)

        if var in col_list:
            return var
        else:
            regex = var + '_[0-9]_[0-9]_[0-9]'
            # regex = '(?<![A-Z])G(?![A-Z])(_[0-9]_[0-9]_[0-9])?'
            cols = utils.get_recols(regex, col_list)
            if cols:
                return cols
            else:
                return None

    def set_data(self):
        self.data = self.flux.copy()
        self._update_var_dict()
    
    def get_highest(self, var):

        # Get columns corresponding to variable
        cols = self.var_dict.get(var)[0]
        # Get columns for variable's highest measurement (VAR_#_1_#)
        regex = self.cls_dict.get(var)[0] + '_[0-9]_1_[0-9]'
        top = utils.get_recols(regex, cols)
        # For one column, return the column
        if isinstance(top, str):
            return self.data[top[0]]
        # For multiple columns, return the mean
        else:
            return self.data[top].mean(axis=1)