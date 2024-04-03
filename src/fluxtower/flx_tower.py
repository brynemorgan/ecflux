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

Name:           flx_tower.py
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
FLX_VAR_INFO_DICT = {}

# FLX_BADM_
def get_ts_var_info(
    meta, rows=['SITE_ID','GROUP_ID'], cols='VARIABLE', vals='DATAVALUE',
    order=[0, 1, -1, 2, 3, 4, 5]
):
    var_info = meta[meta.VARIABLE_GROUP == 'GRP_VAR_INFO'].pivot(
        index=rows, columns=cols, values=vals
    )
    var_info.reset_index(inplace=True)
    var_info = reorder_cols(var_info, order=order)
    var_info.columns = remove_prefix(var_info)
    return var_info


def remove_prefix(df, prefix='VAR_INFO_'):
    cols = [re.sub(r'^{}(.*)'.format(prefix), r'\1', col) for col in df.columns]
    # df.columns = [re.sub(r'^{}(.*)'.format(prefix), r'\1', col) for col in df.columns]
    return cols

def reorder_cols(df, order=[0, -1, 1, 2, 3, 4]):
    return df[[df.columns[i] for i in order]]



class FluxNetTower(FluxTower):
    def __init__(self, filepath, flx_id = None):

        super().__init__(filepath)

        # id
        if flx_id:
            self.id = flx_id
        else:
            self.id = os.path.basename(self._filepath)[4:10]

        # timestep
        self._timestep = self._get_timestep()

        # metadata
        self._all_badm = self._get_all_badm() #self._get_ts_metadata()
        self.set_metadata()
        self.var_info = self._get_var_info()
        self.heights = self.get_heights()

        # flux
        self.flux = self.import_flux()

        # data
        self.set_data()


    
    def set_metadata(self):
        site_badm = self._all_badm[self._all_badm.VARIABLE_GROUP != 'GRP_VAR_INFO']
        self.metadata = { 'ID' : self.id } | utils.cols_to_dict(
            site_badm, key_col='VARIABLE', val_col='DATAVALUE'
        )
        self.metadata.update((k, pd.to_numeric(v,errors='ignore')) for k, v in self.metadata.items())
        
        self.metadata['VEG_HEIGHT'] = self.metadata.get('HEIGHTC', np.nan)

        self._set_coords()
        self._set_tz()

    def _get_all_badm(self):
        all_meta = self._get_ts_metadata()
        return all_meta[all_meta.SITE_ID == self.id]
    
    def _get_ts_metadata(self):
        try:
            return FLX_BADM_DICT[self._timestep]
        except KeyError:
            meta = pd.read_csv(
                os.path.join(os.path.dirname(__file__),'metadata',self._get_meta_file()),
            )
            FLX_BADM_DICT[self._timestep] = meta
            return meta


    def _get_var_info(self, as_dict=True):
        # var_info = self._all_badm[self._all_badm.VARIABLE_GROUP == 'GRP_VAR_INFO']
        # var_info = var_info.pivot(index='GROUP_ID', columns='VARIABLE', values='DATAVALUE')
        # var_info.reset_index(inplace=True)
        # var_info = self.reorder_cols(var_info)
        # var_info.columns = self.remove_prefix(var_info)
        # if as_dict:
        #     var_info = var_info.set_index('VARNAME').to_dict(orient='index')

        try:
            var_info_all = FLX_VAR_INFO_DICT[self._timestep]
        except KeyError:
            var_info_all = get_ts_var_info(self._get_ts_metadata())
            FLX_VAR_INFO_DICT[self._timestep] = var_info_all

        var_info = var_info_all[var_info_all.SITE_ID == self.id]

        if as_dict:
            var_info = var_info.set_index('VARNAME').to_dict(orient='index')
        return var_info

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
        if isinstance(self.var_info, pd.DataFrame):
            return self.var_info.set_index('VARNAME').HEIGHT.to_dict()
        elif isinstance(self.var_info, dict):
            return {k : v['HEIGHT'] for k, v in self.var_info.items()}
    

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
    

    def get_var_cols(self, variable=r'SWC', exclude='QC'):
        # Pattern to match
        pattern = r'{}_(?!.*{})'.format(variable, exclude)
        if isinstance(self.var_info, pd.DataFrame):
            return self.var_info[self.var_info.VARNAME.str.contains(pattern)].VARNAME.tolist()
        elif isinstance(self.var_info, dict):
            return utils.get_recols(pattern, self.var_info.keys())


    def add_vp_cols(self):
        vpd_cols = self.get_var_cols(variable='VPD', exclude='ERA')
        vpd_cols = [col for col in vpd_cols if 'QC' not in col]
        # Calculate saturation vapor pressure
        for col in vpd_cols:
            suff = col.replace('VPD', '')
            self.data['e_star' + suff] = self.calc_svp(suff)
            self.data['e_star' + suff + '_QC'] = self.data['TA' + suff + '_QC'].copy()
            self.data['e_a' + suff] = self.calc_vp(self.data['e_star' + suff], suff)
            self.data['e_a' + suff + '_QC'] = self.data['VPD' + suff + '_QC'].copy()

            self._update_var_info(var='e_star'+suff, unit='kPa', height_var='TA'+suff)
            self._update_var_info(var='e_a'+suff, unit='kPa', height_var='VPD'+suff)
        # return self

    def calc_vp(self, svp, suff = '_F_MDS'):

        e_a = svp - (self.data['VPD' + suff]*0.1)
        e_a.name = 'e_a' + suff

        return e_a

    def calc_svp(self, suff='_F_MDS'):
        # Calculate saturation vapor pressure
        svp = 0.611 * np.exp((17.502 * self.data['TA' + suff]) / (self.data['TA' + suff] + 240.97))

        svp.name = 'e_star' + suff

        return svp

    def add_et_cols(self, suff='_F_MDS'):
        self.data['ET' + suff] = self.calc_et(suff)
        self.data['ET' + suff + '_QC'] = self.data['LE' + suff + '_QC'].copy()
        if self._timestep == 'HH':
            unit = 'mm hr-1'
        else:
            unit = 'mm d-1'
        
        self._update_var_info(var='ET'+suff, unit=unit, height_var='LE' + suff)

    def calc_et(self, suff='_F_MDS'):

        lambda_v = (2.501 - (2.361e-3 * self.data['TA_F_MDS'])) * 1e6

        if self._timestep == 'HH':
            mult = 3600     # mm hr-1
        else:
            mult = 86400    # mm day-1
        et = (self.data['LE' + suff] * mult) / lambda_v  

        return et

    def _update_var_info(self, var, unit, height_var=None):
        if not height_var:
            h = np.nan
        else:
            h = self.var_info[height_var]['HEIGHT']

        new_col = {
            'HEIGHT' : h,
            'MODEL' : 'calculated',
            'UNIT' : unit,
        }
        qc_col  = {
            'HEIGHT' : np.nan,
            'MODEL' : 'calculated',
            'UNIT' : 'nondimensional',
        }
        
        # TODO: Make this work when var_info is a df (currently only works for dict)
        
        self.var_info[var] = new_col
        self.var_info[var + '_QC'] = qc_col
