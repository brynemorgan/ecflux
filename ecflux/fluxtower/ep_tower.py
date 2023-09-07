#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Sat 20 Aug 22 00:03:13'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           ep_tower.py
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

from fluxtower import FluxTower
from .FFP_Python import calc_footprint_FFP as calc_ffp
from .FFP_Python import calc_footprint_FFP_climatology as calc_ffp_clim
from .footprint import Footprint
# from . import ffp, ffp_clim
# from fluxtower.utils import get_recols,import_dat


# FUNCTIONS

class EddyProTower(FluxTower):

    def __init__(self, filepath, meta_file, biomet_files=None):

        super().__init__(filepath, meta_file, biomet_files)

        # _flux_files
        self._flux_files = self.get_flux_files()

        # flux
        # self.flux = pd.concat(
        #     [self.import_flux(
        #         os.path.join(self._filepath,'summaries',file), index_col=0
        #     ) for file in self._flux_files]
        # )
        self.flux = self.import_flux()

        # data
        self.set_data()
        self.clean_data()
        self.mask_data()
    
    def get_flux_files(self):

        files = [ txt for txt in sorted(
            os.listdir(os.path.join(self._filepath, 'summaries'))
        ) if txt[0] != '.' ]

        return files

    def import_flux(self):
        # Import all flux files
        flux = pd.concat(
            [self._import_flux(
                os.path.join(self._filepath,'summaries',file)
            ) for file in self._flux_files],
            ignore_index=True
        )

        # Set timezone for datetime column
        dt_named = self._set_col_tz(flux.date_time)
        # Set tz-aware datetime index
        flux.set_index(dt_named, inplace=True)

        return flux



    @staticmethod
    def _import_flux(file, skiprows=[1], parse_dates=[[2,3]], delim_whitespace=True, **kwargs):
        """
        Import 30-min flux data from EddyPro output file.

        Parameters
        ----------
        file : str
            The name of the file to be imported.

        skiprows : list-like, int
            Rows of table to skip; passed to pd.read_csv(). The default is [1] (the row with the units).

        parse_dates : bool, list of ints, or list of lists
            Columns of file to parse into datetime objects; passed to pd.read_csv().
            The default is [[2,3]], which parses the date and time columns into a
            single datetime column. This should not need to be changed.

        delim_whitespace : bool
            Specifies whether or not to interpret whitespace as the delimiter between
            columns; passed to pd.read_csv(). Equivalent to setting sep='\s+'. This
            parameter is magical. The default is True.

        **kwargs
            kwargs to be passed to pd.read_csv(). Options can be found in the documentation
            (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)

        Returns
        -------
        df : DataFrame
            Contains data imported from file.
        """

        df = pd.read_csv(
            file, skiprows=skiprows, parse_dates=parse_dates, 
            delim_whitespace=delim_whitespace, **kwargs
        )

        # df.date_time = df.date_time.apply(utils.make_tzaware, utc_offset=-8, tz_name='America/Los_Angeles')

        return df
    
    def ffp(self, timestamp=None, ffp_dict=None, clim=True, avg=False, 
            use_umean=True, wind_dir=True, rs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 
            domain_arr=None, use_res=False, nx=1000, ny=None, dx=None, dy=None, 
            fig=False, **kwargs):

        # Raise an error if neither timestamp nor ffp_dict were passed.
        if not timestamp and not ffp_dict:
            raise TypeError("ffp() missing 1 of 2 required arguments: 'timestamp' or 'ffp_dict'") 

        # Get the params for ffp if not provided
        if not ffp_dict:
            ffp_dict = self.get_ffp_params(timestamp, avg=avg)

        # Copy ffp_dict
        ffp_input = ffp_dict.copy()
        
        # If desired, pass wind speed rather than roughness length (set z0 to None).
        if use_umean:
            ffp_input.update({'z0' : None})
        
        # Run footprint calculation
        if clim:
            # If an array was passed, get the extent to pass to calc_FFP_climatology()
            if domain_arr is not None:
                domain = self.get_domain(domain_arr)
            else:
                domain = None
                # if use_res:
                #     dx = domain_arr.rio.resolution()
            # Run footprint calculation
            footprint = calc_ffp_clim.FFP_climatology(
                **ffp_input, rs=rs, domain=domain, nx=nx, ny=ny, dx=dx, dy=dy, fig=fig, **kwargs
            )
        # Run regular footprint calculation
        else:
            # Don't use wind direction (requires z_0m)
            if not wind_dir:
                ffp_input.update({'wind_dir' : None})
            # Run footprint calculation
            footprint = calc_ffp.FFP(**ffp_input, rs=rs, nx=nx, fig=fig, **kwargs)

        return footprint



    def calc_footprint(self, timestamp=None, ffp_dict=None, clim=True, avg=False, 
                       use_umean=True, wind_dir=True, rs=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9], 
                       domain_arr=None, use_res=False, nx=1000, ny=None, dx=None, dy=None, 
                       fig=False, **kwargs) -> Footprint :

        # Raise an error if neither timestamp nor ffp_dict were passed.
        if timestamp is None and not ffp_dict:
            raise TypeError("ffp() missing 1 of 2 required arguments: 'timestamp' or 'ffp_dict'") 

        # Get the params for ffp if not provided
        if not ffp_dict:
            ffp_dict = self.get_ffp_params(timestamp, avg=avg)

        # Copy ffp_dict
        ffp_input = ffp_dict.copy()
        
        # If desired, pass wind speed rather than roughness length (set z0 to None).
        if use_umean:
            ffp_input.update({'z0' : None})
        
        # If an array was passed, get the extent to pass to calc_FFP_climatology()
        if domain_arr is not None:
            domain = self.get_domain(domain_arr)
        else:
            domain = None
            # if use_res:
            #     dx = domain_arr.rio.resolution()
        footprint = Footprint(
            ffp_input, timestamp=timestamp, coords=self.coords,
            rs=rs, domain=domain, nx=nx, dx=dx, dy=dy, fig=fig, **kwargs
        )
        return footprint



    def get_ffp_params(self, timestamp = None, avg=False):
        """
        Create a dictionary of parameters to pass to ffp.

        Parameters
        ----------
        timestamp : pd.Timestamp | array-like, optional
            Timestamp(s) for which to generate footprint. If a single timestamp
            is passed, a dictionary with scalar values will be returned. If multiple
            timestamps are passed, the dictionary will have array-like values. 
            The default is to return all values from self.data.
        avg : bool, optional
            Whether or not an average footprint will be calculated over the timestamps.
            If timestamp is a single value, this is ignored. Otherwise, if avg is True,
            the dictionary will be returned and is intended to be passed to ffp. 
            If False, a dict of dicts will be returned, with each timestamp having
            a dict of params. The default is False.

        Returns
        -------
        ffp_params : dict
            A dictionary containing either: variables as keys or (if timestamp
            is iterable and avg is False), timestamps as keys and variables as keys
            in the value dict.
        """
        if timestamp is None:
            ffp_params = {
                'zm' : self.z - self.d_0,
                'z0' : self.z_0m,
                'umean' : self.data.u,
                'h' : self.calc_abl_height(self.data.L, self.data.ustar),
                'ol' : self.data.L,
                'sigmav' : np.sqrt(self.data.v_var),
                'ustar' : self.data.ustar,
                'wind_dir' : self.data.wind_dir
            }
        else:
            ffp_params = {
                'zm' : self.z - self.d_0,
                'z0' : self.z_0m,
                'umean' : self.data.u[timestamp],
                'h' : self.calc_abl_height(self.data.L[timestamp], self.data.ustar[timestamp]),
                'ol' : self.data.L[timestamp],
                'sigmav' : np.sqrt(self.data.v_var[timestamp]),
                'ustar' : self.data.ustar[timestamp],
                'wind_dir' : self.data.wind_dir[timestamp]
            }
        
        if not avg and hasattr(timestamp, '__iter__'):
            ffp_params = pd.DataFrame(ffp_params).to_dict(orient='index')

        return ffp_params


    def calc_abl_height(self, L, u_star):
        """
        Calculates the height of the boundary layer.

        Parameters
        ----------
        L : float or array-like
            Obukhov length [m]
            Length must match length of u_star.

        u_star : float or array-like
            Friction velocity [m s-1]
            Length must match length of L.

        lat : float
            Latitude in decimal degrees.

        Returns
        -------
        h : float or array-like
            Height of the ABL [m].
        """
        omega = 7.2921e-5   # angular velocity of the earth's rotation [rad s-1]
        
        # For convective conditions, set h = 1500
        if np.isscalar(L):
            if L < 0:
                h = 1500
        else:
            # Calculate coriolis parameter [s-1] (https://en.wikipedia.org/wiki/Coriolis_frequency)
            f = 2 * omega * np.sin(np.radians(self.coords[0]))

            # Calculate h
            h = (L / 3.8) * (-1 + np.sqrt(1 + 2.28 * (u_star / (f * L) )))

        # For convective conditions, set h = 1500
        if isinstance(h, pd.core.series.Series):
            h = h.fillna(1500)

        return h

    def flag_data(self,):

        flag_dict = {
            'F_RAIN' : 11,
            'F_LE' : 12,
            'F_SW' : 13,
            'F_EBR' : 1
        }

        self.data['FLAG'] = self.get_qc_flag()
        self.data.FLAG[(self.data.FLAG == 0) & (self.data.P_RAIN_1_1_1 != 0)] = flag_dict.get('F_RAIN')
        self.data.FLAG[(self.data.FLAG == 0) & (self.data.LE.isna())] = flag_dict.get('F_LE')
        self.data.FLAG[(self.data.FLAG == 0) & (self.data.SW_IN < -10)] = flag_dict.get('F_SW')
        # self.data.FLAG[(self.data.FLAG == 0) & (self.data.EBR_perc > 0.20)] = flag_dict.get('F_EBR')


    def get_qc_flag(self):

        qc_flag = self.data.qc_Tau * 100 + self.data.qc_H * 10 + self.data.qc_LE

        return qc_flag

    def mask_data(self):

        sw_mask = self.data.SW_IN < -10.
        self.fill_na(sw_mask, cols=['SW_IN', 'R_n'])


    def fill_na(self, mask, cols):

        raw_cols = [col + '_raw' for col in cols]
        self.data[raw_cols] = self.data[cols].copy()
        self.data[cols] = self.data.where(~mask, np.nan)[cols]

