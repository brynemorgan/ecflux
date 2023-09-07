#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Tue 04 Oct 22 11:17:28'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           footprint.py
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
import numpy as np
import xarray as xr

from fluxtower import utils

from .FFP_Python import calc_footprint_FFP as calc_ffp
from .FFP_Python import calc_footprint_FFP_climatology as calc_ffp_clim

# CLASS

class Footprint:

    def __init__(self, in_params : dict, timestamp, coords : tuple = None, **kwargs):

        self._params = in_params
        self.timestamp = timestamp
        self.origin = coords

        self.ffp = self.calc_footprint(**kwargs)
        self.set_footprint_vals()



    def calc_footprint(self, **kwargs):

        ffp_out = calc_ffp_clim.FFP_climatology(**self._params, **kwargs)

        return ffp_out

    def set_footprint_vals(self):

        self.x_coords = self.ffp['x_2d'][0]
        self.y_coords = self.ffp['y_2d'][:,0]
        
        self.footprint = self.ffp['fclim_2d']

        if self.origin:
            self.set_footprint_utm()

    def set_footprint_utm(self):

        # Convert origin to UTM coordinates
        self.origin_utm = self.get_utm_coords()
        # Create data array with UTM coordinates
        self.foot_utm = xr.DataArray(
            self.footprint, 
            coords = (self.y_coords + self.origin_utm[1], self.x_coords + self.origin_utm[0]),
            dims = ['y', 'x']
        )
        # Set CRS
        crs = utils.get_utm_crs(lat=self.origin[0], long=self.origin[1])
        self.foot_utm.rio.set_crs(crs, inplace=True)
        # Set resolution
        self.resolution = self.foot_utm.rio.resolution()
        self.pixel_area = self.resolution[0] * self.resolution[1]
        # self.footprint_fraction = (self.foot_utm * self.pixel_area).sum().item()
        self.footprint_fraction = self.calc_footprint_fraction()
        # Get contour lines in UTM coordinates
        self.contours = self.get_contour_points(self.origin_utm)


    def get_utm_coords(self, datum='WGS 84'):
        coords_utm = utils.convert_to_utm(
            self.origin[0], self.origin[1], datum=datum
        )
        return coords_utm

    def get_contour_points(self, utm_coords : tuple):
        """
        Get the contour lines of the footprint as tuples. Can be either in relative 
        coordinates (with tower at (0,0)) or UTM coordinates if passed.

        Parameters
        ----------
        footprint : dict
            footprint model (output of calc_footprint_FFP_climatology or calc_footprint_FFP)
            
        utm_coords : tuple, optional
            UTM coordinates of tower, by default ramajal.get_utm_coords()

        Returns
        -------
        rs_dict : dict
            Dictionary with contour levels (fraction of source area) as keys and lists
            of tuples as values.
        """
        # Convert to UTM coordinates
        if utm_coords:
            x_utm = [np.array(x) + utm_coords[0] for x in self.ffp['xr'] if x]
            y_utm = [np.array(y) + utm_coords[1] for y in self.ffp['yr'] if y]

            rs_dict = {r : list(zip(x,y)) for (r,x,y) in zip(self.ffp['rs'],x_utm,y_utm) if x is not None}
        # Just return distances (relative coordinates with tower at (0,0))
        else:
            rs_dict = {r : list(zip(x,y)) for (r,x,y) in zip(self.ffp['rs'],self.ffp['xr'],self.ffp['yr']) if x is not None}

        return rs_dict

    def calc_footprint_fraction(self, arr : xr.DataArray = None):
        """
        _summary_

        Parameters
        ----------
        arr : xr.DataArray
            Input array to use as mask. NaNs in this array will not be included 
            in the total footprint.
        """
        if arr is None:
            foot = self.foot_utm
        else:
            # Resample
            # arr = arr.rio.reproject(self.foot_utm.rio.crs, resolution=self.resolution)
            arr_resamp = arr.interp(x=self.foot_utm.x, y=self.foot_utm.y)
            foot = self.foot_utm.where(arr_resamp.notnull())
        
        return (foot * self.pixel_area).sum().item()