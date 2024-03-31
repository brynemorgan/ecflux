#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2024'

__license__ = 'MIT'
__date__ = 'Tue 26 Mar 24 20:49:14'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           fluxnet_tower.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2024


"""


# IMPORTS
import os
import numpy as np
import pandas as pd
import re

from fluxtower import FluxTower,utils



# VARIABLES
META = pd.read_csv(
    os.path.join(
        os.path.dirname(__file__),
        'metadata',
        'FLX_AA-Flx_BIF_DD_20200501.csv'
    )
)


FLX_BADM_DICT = {}

# FLX_BADM_


class FluxNetTower(FluxTower):
    def __init__(self, filepath, flx_id = None):

        super().__init__(filepath)

        # flx_id
        if flx_id:
            self.flx_id = flx_id
        else:
            self.flx_id = os.path.basename(self._filepath)[4:10]

        # timestep
        self._timestep = self._get_timestep()

        # metadata
        self._all_badm = self.get_all_badm() #self._get_ts_metadata()
        self.set_metadata()
        self._var_info = self.get_var_info()
        self.heights = self.get_heights()

        # flux
        self.flux = self.import_flux()

        # data
        self.set_data()


    def _get_ts_metadata(self):
        try:
            return FLX_BADM_DICT[self._timestep]
        except KeyError:
            meta = pd.read_csv(
                os.path.join(os.path.dirname(__file__),'metadata',self._get_meta_file()),
            )
            FLX_BADM_DICT[self._timestep] = meta
            return meta
    
    def get_all_badm(self):
        all_meta = self._get_ts_metadata()
        return all_meta[all_meta.SITE_ID == self.flx_id]
    
    def set_metadata(self):
        site_badm = self._all_badm[self._all_badm.VARIABLE_GROUP != 'GRP_VAR_INFO']
        self.metadata = { 'ID' : self.flx_id } | utils.cols_to_dict(
            site_badm, key_col='VARIABLE', val_col='DATAVALUE'
        )
        self.metadata.update((k, pd.to_numeric(v,errors='ignore')) for k, v in self.metadata.items())

        self._set_coords()
        self._set_tz()

    # def _get_all_badm(self):
    #     return AMF_BADM_DF[AMF_BADM_DF.SITE_ID == self.amf_id]


    def _get_meta_file(self, date='20200501'):
        return f'FLX_AA-Flx_BIF_{self._timestep}_{date}.csv'


    def _get_timestep(self):
        # pattern = r'_BIF_([A-Za-z]+)_\d{8}\.csv$'
        pattern = r'SET_([A-Za-z0-9]+)_'
        match = re.search(pattern, os.path.basename(self._filepath))
        if match:
            return match.group(1)
        else:
            return None
        
    def get_heights(self):
        if isinstance(self._var_info, pd.DataFrame):
            return self._var_info.set_index('VARNAME').HEIGHT.to_dict()
        elif isinstance(self._var_info, dict):
            return {k : v['HEIGHT'] for k, v in self._var_info.items()}


    # var_groups = meta.groupby('VARIABLE_GROUP').VARIABLE.unique().apply(list).to_dict()
    def get_var_info(self, as_dict=True):
        var_info = self._all_badm[self._all_badm.VARIABLE_GROUP == 'GRP_VAR_INFO']
        var_info = var_info.pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
        var_info.reset_index(inplace=True)
        var_info = self.reorder_cols(var_info)
        var_info.columns = self.remove_prefix(var_info)
        if as_dict:
            var_info = var_info.set_index('GROUP_ID').to_dict(orient='index')

        return var_info
    

    def remove_prefix(self, df, prefix='VAR_INFO_'):
        cols = [re.sub(r'^{}(.*)'.format(prefix), r'\1', col) for col in df.columns]
        # df.columns = [re.sub(r'^{}(.*)'.format(prefix), r'\1', col) for col in df.columns]
        return cols

    def reorder_cols(self, df, order=[0, -1, 1, 2, 3, 4]):
        return df[[df.columns[i] for i in order]]

    def _set_coords(self):

        # self.lat = float(self.metadata.get('LAT'))
        # self.lon = float(self.metadata.get('LONG'))
        self.coords = (float(self.metadata.get('LOCATION_LAT')), float(self.metadata.get('LOCATION_LONG')))
        self.alt = self.metadata.get('LOCATION_ELEV')
        if self.alt:
            self.alt = float(self.alt)


    def import_flux(self, skiprows=0, na_values=-9999., **kwargs):

        '''
        NOTE: Would like to parse dates on import (see below), but this is ~9x slower than
        converting dates afterwards...
        flux = pd.read_csv(base_file, skiprows=2, parse_dates=[0,1], infer_datetime_format=True, na_values=-9999.)

        '''

        flux = pd.read_csv(self._filepath, skiprows=skiprows, na_values=na_values, **kwargs)

        ts_cols = utils.get_recols('TIMESTAMP', flux.columns)

        for col in ts_cols:
            try:
                flux[col] = pd.to_datetime(flux[col], format='%Y%m%d%H%M')
            except:
                flux[col] = pd.to_datetime(flux[col], format='%Y%m%d')

        # dt_named = self._set_col_tz(flux.TIMESTAMP_END)

        dt_named = self._set_col_tz(flux[col])  # col will be 'TIMESTAMP' or 'TIMESTAMP_END'

        flux.set_index(dt_named, inplace=True)

        return flux

    def set_data(self):
        self.data = self.flux.copy()
        # self._update_var_dict()
    

