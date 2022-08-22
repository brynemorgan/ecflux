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


def import_dat(file, skiprows=[0,2,3], na_values='NAN', index_col='TIMESTAMP', 
            parse_dates=True, **kwargs):

    dat = pd.read_csv(
        file, skiprows=skiprows, na_values=na_values, index_col=index_col, 
        parse_dates=parse_dates, **kwargs
    )

    return dat


def convert_units(val, unit):

    # TODO: Don't do it this way. Refactor maybe with dict to map to various conversions?

    if unit == 'C':
        return val + 273.15
    elif unit == 'Pa':
        return val / 1000
    elif unit == 'm':
        return val / 1000