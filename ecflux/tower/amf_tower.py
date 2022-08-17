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

from tower import FluxTower

# VARIABLES

AMF_META_VARS = {
    'SITE_ID' : 'SITE_ID',
    'SITE_NAME' : 'SITE_NAME',
    'DATE_START' : 'FLUX_MEASUREMENTS_DATE_START',
    'DATE_END' : 'FLUX_MEASUREMENTS_DATE_END',
    'CLIMATE_MAT' : 'MAT',
    'CLIMATE_MAP' : 'MAP',
    'CLIMATE_KOEPPEN' : 'CLIMATE_KOEPPEN',
    'VEG_IGBP' : 'IGBP',
    'LAT' : 'LOCATION_LAT',
    'LONG' : 'LOCATION_LONG',
    'ELEV' : 'LOCATION_ELEV',
    'UTC_OFFSET' : 'UTC_OFFSET',
    'VEG_HEIGHT' : 'HEIGHTC',
}

# Numeric columns (incomplete list)
num_cols = ['MAT', 'MAP', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV', 'UTC_OFFSET', 'HEIGHTC']

def cols_to_dict(df, key_col='VARIABLE', val_col='DATAVALUE'):

    return pd.Series(df[val_col].values,index=df[key_col]).to_dict()

# Raw AMF BIF data
AMF_METADATA_DF = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'AMF_AA-Net_BIF_CCBY4_20220811.csv'
    )
)# AMF BADM data for all sites (dict of dicts)
AMF_SITE_METADATA = AMF_METADATA_DF.groupby('SITE_ID').apply(cols_to_dict).to_dict()

# Restructured AMF BADM data
AMF_METADATA_DB = pd.DataFrame(list(AMF_METADATA_DF.groupby('SITE_ID').apply(cols_to_dict).values))
AMF_METADATA_DB.insert(0, 'SITE_ID',AMF_METADATA_DF.SITE_ID.unique())
# Convert columns to numeric
AMF_METADATA_DB[num_cols] = AMF_METADATA_DB[num_cols].apply(pd.to_numeric, errors='coerce')



class AmeriFluxTower(FluxTower):

    def __init__(self, amf_id, filepath):

        super().init(filepath)

        self.amf_id = amf_id

        self.badm = self.get_badm(self.amf_id)

    
    def set_metadata(self):
        
        return None
    
    def get_badm(self, amf_id):

        return AMF_METADATA_DB[AMF_METADATA_DB.SITE_ID == amf_id].to_dict(orient='records')

