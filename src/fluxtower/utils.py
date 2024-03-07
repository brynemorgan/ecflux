#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Sat 20 Aug 22 00:06:56'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           utils.py
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
import re
import pandas as pd
import pyproj


# VARIABLES

var_df = pd.read_csv(
    os.path.join(os.path.dirname(os.path.realpath(__file__)),'variables.csv'), 
    index_col='VARIABLE'
)
var_df = var_df.where(pd.notnull(var_df), None)

VARIABLES = var_df['Variable_Units'].to_dict()


# FUNCTIONS

def get_var_dict(subclass):

    var_dict = dict( zip( 
        var_df.index, 
        zip( 
            var_df[subclass[:-5] + '_Name'].values, 
            var_df[subclass[:-5] + '_Units'].values 
        ) 
    ) )

    return var_dict


def get_recols(regex, col_list):

    r = re.compile(regex)

    cols = list(filter(r.match, col_list))

    return cols


# def import_dat(file, skiprows=[0,2,3], na_values='NAN', index_col='TIMESTAMP', 
#             parse_dates=True, **kwargs):

#     dat = pd.read_csv(
#         file, skiprows=skiprows, na_values=na_values, index_col=index_col, 
#         parse_dates=parse_dates, **kwargs
#     )

#     return dat


def convert_units(val, unit):

    # TODO: Don't do it this way. Refactor maybe with dict to map to various conversions?

    if unit == 'C':
        return val + 273.15
    elif unit == 'Pa':
        return val / 1000
    elif unit == 'm':
        return val / 1000


def get_utm_crs(lat, long, datum='WGS 84'):

    aoi = pyproj.aoi.AreaOfInterest(
        west_lon_degree = long,
        south_lat_degree = lat,
        east_lon_degree = long,
        north_lat_degree = lat
    )

    utm_crs_list = pyproj.database.query_utm_crs_info(datum, aoi)

    utm_crs = pyproj.CRS.from_epsg(utm_crs_list[0].code)

    return utm_crs

def convert_to_utm(lat, long, datum='WGS 84'):

    # Get UTM CRS
    utm_crs = get_utm_crs(lat, long, datum=datum)
    # Create transformer
    proj = pyproj.Transformer.from_crs(4326, utm_crs)
    # Transform
    coords_utm = proj.transform(lat, long)

    return coords_utm


def calc_ebr(R_n, H, LE, G):

    ebr = R_n - G - H - LE

    return ebr

def attribute_ebr_bowen(H, LE, EBR):

    beta = H/LE

    H_corr = H + ( (EBR * beta) / (1 + beta) )
    LE_corr = LE + ( EBR / (1 + beta) )

    return H_corr, LE_corr

def attribute_ebr_h(H, LE, EBR):

    H_corr = H + EBR

    return H_corr, LE

def attribute_ebr_le(H, LE, EBR):

    LE_corr = LE + EBR

    return H, LE_corr


ebr_dict = {
    'bowen' : attribute_ebr_bowen,
    'H' : attribute_ebr_h,
    'LE' : attribute_ebr_le
}

def attribute_ebr(H, LE, EBR, method='bowen'):

    ebr_method = ebr_dict.get(method, 'bowen')

    return ebr_method(H, LE, EBR)