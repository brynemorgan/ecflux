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

from fluxtower import FluxTower

# VARIABLES

AMF_META_VARS = {
    'AMF_ID' : 'AMF_ID',
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
        'TOWER_HEIGHT' : None
    },
    'US-xBR' : {
        'SITE_ID' : 'BART',
        'VEG_HEIGHT' : 23.0,
        'TOWER_HEIGHT' : None
    },
    'US-xGR' : {
        'SITE_ID' : 'GRSM',
        'VEG_HEIGHT' : 30.0,
        'TOWER_HEIGHT' : None
    },
    'US-xHA' : {
        'SITE_ID' : 'HARV',
        'VEG_HEIGHT' : 26.0,
        'TOWER_HEIGHT' : None
    },
    'US-xRN' : {
        'SITE_ID' : 'ORNL',
        'VEG_HEIGHT' : 28.0,
        'TOWER_HEIGHT' : None
    },
    'US-xSE' : {
        'SITE_ID' : 'SERC',
        'VEG_HEIGHT' : 38.0,
        'TOWER_HEIGHT' : None
    },
    'US-xSJ' : {
        'SITE_ID' : 'SJER',
        'VEG_HEIGHT' : 21.0,
        'TOWER_HEIGHT' : 39.0
    },
}

# Numeric columns (incomplete list)
num_cols = ['MAT', 'MAP', 'LOCATION_LAT', 'LOCATION_LONG', 'LOCATION_ELEV', 'UTC_OFFSET', 'HEIGHTC']

def cols_to_dict(df, key_col='VARIABLE', val_col='DATAVALUE'):

    return pd.Series(df[val_col].values,index=df[key_col]).to_dict()

# Raw AMF BIF data
AMF_BADM_DF = pd.read_csv(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)), 
        'AMF_AA-Net_BIF_CCBY4_20220811.csv'
    )
)# AMF BADM data for all sites (dict of dicts)
AMF_SITE_BADM = AMF_BADM_DF.groupby('SITE_ID').apply(cols_to_dict).to_dict()

# Restructured AMF BADM data
AMF_BADM_DB = pd.DataFrame(list(AMF_BADM_DF.groupby('SITE_ID').apply(cols_to_dict).values))
AMF_BADM_DB.insert(0, 'SITE_ID',AMF_BADM_DF.SITE_ID.unique())
# Convert columns to numeric
AMF_BADM_DB[num_cols] = AMF_BADM_DB[num_cols].apply(pd.to_numeric, errors='coerce')

# AMF_SUPP_META = pd.read_csv(
#     os.path.join(
#         os.path.dirname(os.path.realpath(__file__)), 
#         'AMF_Supplementary_Metadata.csv'
#     )
# )


class AmeriFluxTower(FluxTower):

    def __init__(self, filepath, amf_id=None):

        super().__init__(filepath)

        # amf_id
        if amf_id:
            self.amf_id = amf_id
        else:
            self.amf_id = os.path.basename(self._filepath)[4:10]

        # badm
        self.badm = {'AMF_ID' : self.amf_id} | AMF_SITE_BADM.get(self.amf_id)
        # self.badm = AMF_BADM_DB[AMF_BADM_DB.SITE_ID == self.amf_id].to_dict(orient='records')
        
        # metadata
        self.set_metadata()

    
        

    def set_metadata(self):

        self.metadata = {k : self.badm.get(v, None) for k,v in AMF_META_VARS.items()}
        self.metadata.update(AMF_SUPP_META.get(self.amf_id))
        

