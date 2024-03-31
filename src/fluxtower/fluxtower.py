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
import pandas as pd
import datetime
import pytz
import timezonefinder as tzf

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

        self._set_coords()
        self._set_tz()
        self._set_height_params()

    # def import_flux(self, file, **kwargs):

    #     flux = pd.read_csv(file, **kwargs)
    #     return flux
    def import_flux(self):
        raise NotImplementedError
    
    def import_dat(self, file, skiprows=[0,2,3], na_values='NAN', dt_col='TIMESTAMP', 
            parse_dates=True, **kwargs):

        # Import data from .dat file
        dat = pd.read_csv(
            file, skiprows=skiprows, na_values=na_values, 
            parse_dates=parse_dates, **kwargs
        )
        # Convert dt_col to datetime
        dat[dt_col] = pd.to_datetime(dat[dt_col])
        # Set timezone for datetime column
        dt_named = self._set_col_tz(dat[dt_col])
        # Set tz-aware datetime index
        dat.set_index(dt_named, inplace=True)

        return dat
    
    def set_biomet(self, biomet_files):

        if isinstance(biomet_files, str):
            self.biomet = self.import_dat(biomet_files)
        elif isinstance(biomet_files, list):
            self.biomet_list = [ self.import_dat(file) for file in biomet_files ]
            self.biomet = self.biomet_list[0].join(self.biomet_list[1:], how='inner')

    def _set_coords(self):

        # self.lat = float(self.metadata.get('LAT'))
        # self.lon = float(self.metadata.get('LONG'))
        self.coords = (float(self.metadata.get('LAT')), float(self.metadata.get('LONG')))
        self.alt = float(self.metadata.get('ELEV'))

    
    def _set_tz(self):
        # Get fixed UTC offset [hours] (no DST)
        self._utc_offset = int(self.metadata.get('UTC_OFFSET'))
        # Get named timezone 
        self._tz_name = tzf.TimezoneFinder().timezone_at(
            lng=self.coords[1], lat=self.coords[0]
        )

    def _set_col_tz(self, dt_naive):

        # Get timezone based on UTC offset (no DST)
        # utc_offset = self.metadata.get('UTC_OFFSET')
        tz_offset = datetime.timezone(datetime.timedelta(hours=self._utc_offset))
        # Set fixed UTC-offset timezone for DateTimeIndex 
        dt_utcoff = dt_naive.dt.tz_localize(tz = tz_offset)
        # Next, get named timezone as pytz.timezone object from tz name
        tz_named = pytz.timezone(self._tz_name)
        # Convert timezone to named timezone (for general compatibility)
        dt_named = dt_utcoff.dt.tz_convert(tz = tz_named)

        return dt_named

    def get_utm_coords(self, datum='WGS 84'):
        coords_utm = utils.convert_to_utm(
            self.coords[0], self.coords[1], datum=datum
        )
        return coords_utm

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
        # Calculate net radiation if necessary
        if 'R_n' not in self.data.columns:
            self.calc_Rn()
        # Calculate available energy
        self.data['Q_av'] = self.data.R_n - self.data.G
        # Calculate energy balance residual
        self.data['EBR'] = utils.calc_ebr(self.data.R_n, self.data.H, self.data.LE, self.data.G)

    def calc_Rn(self):

        self.data['R_n'] = self.data.SW_IN - self.data.SW_OUT + self.data.LW_IN - self.data.LW_OUT

    def attribute_ebr(self, method='all'):

        if method == 'all':
            for meth in utils.ebr_dict.keys():
                H_corr, LE_corr = utils.attribute_ebr(
                    self.data.H, self.data.LE, self.data.EBR, method=meth
                )
                self.data['H_corr_' + meth] = H_corr
                self.data['LE_corr_' + meth] = LE_corr
        else:
            H_corr, LE_corr = utils.attribute_ebr(
                self.data.H, self.data.LE, self.data.EBR, method=method
            )
            self.data['H_corr_' + method] = H_corr
            self.data['LE_corr_' + method] = LE_corr


    def _set_height_params(self):

        # Tower height, z
        self.z = self._get_tower_height()
        # Zero-plane displacement height, d0
        self.d_0 = self._calc_d0()
        # Roughness length, z_0m
        self.z_0m = self._calc_z0m()


    def _get_tower_height(self):
        return float(self.metadata.get('TOWER_HEIGHT'))
    
    def _calc_d0(self):
        """
        Calculate the zero-plane displacement height (height at which wind speed goes
        to 0), d_0 [m].

        Parameters
        ----------
        h : float
            Canopy height [m].

        Returns
        -------
        d_0 : float
            Zero-plane displacement height [m].

        Reference: Norman et al. (1995).
        """
        d_0 = 0.65 * self.metadata.get('VEG_HEIGHT')

        return d_0

    def _calc_z0m(self):
        """
        Calculate the aerodynamic roughness length for momemtum transport, z_0m [m].

        Parameters
        ----------
        h : float
            Canopy height [m]

        Returns
        -------
        z_0m : float
            Roughness length for momentum transport [m].

        Reference: Norman et al. (1995).
        """
        z_0m = 0.125 * self.metadata.get('VEG_HEIGHT')

        return z_0m
    
    def get_domain(self, arr_in):

        arr = arr_in.dropna(dim='x', how='all')
        arr = arr.dropna(dim='y', how='all')

        utm_coords = self.get_utm_coords()
        # Get distance from point
        x_dist = arr.x - utm_coords[0]
        y_dist = arr.y - utm_coords[1]
        # Get max and min distances in x and y
        x_min = x_dist.min().item()
        x_max = x_dist.max().item()
        y_min = y_dist.min().item()
        y_max = y_dist.max().item()
        # Create list for domain
        domain = [x_min, x_max, y_min, y_max]

        return domain