#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Fri 21 May 21 10:43:23'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           tower.py
Compatibility:  Python 3.7.0
Description:    Description of what program does

URL:            https://

Requires:       list of libraries required

Dev ToDo:       None

AUTHOR:         Bryn Morgan
ORGANIZATION:   University of California, Santa Barbara
Contact:        bryn.morgan@geog.ucsb.edu
Copyright:      (c) Bryn Morgan 2021


"""


#%% IMPORTS

import os
import numpy as np
import pandas as pd

from uavet import utils


#%%

# THINGS TO PUT IN A CONFIG:
config = {
    'dir': '~/Box/Dangermond/RamajalTower',
    'raw_dir': '~/Box/Dangermond/RamajalTower/raw_unzipped/data/',
    'biomet_dir': '~/Box/Dangermond/RamajalTower/raw_unzipped/biomet/',
    'dt_format': '%Y-%m-%dT%H%M%S',
    'utc_offset' : -8,
    'tz': 'America/Los_Angeles',
    'skip_char' : 0,
    'end_char': 13
}


#%%

class FluxTower():
    """

    """

    def __init__(self, filepath=None, meta=None):

        
        # config
        if config:
            self.config = config


        # filename
        self.filename = filename
        # tz
        self.tz = config.get('tz', 'America/Los_Angeles')

        # Option 1. Pass a filename.
        #if filename:

        # Option 2. Pass a flux_data object.
        if flux_data:
            self.filename = flux_data.filename

        # Option 3. Pass a flux_bucket object.


        # Option 4. Pass a config.

        # Attributes/things the FluxTower class should "hold":
        #   + 30-min flux data (DataFrame)
        #       for smartflux/eddypro towers, this file includes most biomet data (except irts)
        #   + raw 10-Hz data (DataFrame, for some time period?)
        #   + raw biomet data (DataFrame, for some time period?)

        # Methods/things the class should "do":
        #   + footprint stuff
        #   + get tower data for drone flight (i.e. pass timestamp and get closest half-hour)





class EddyProTower(FluxTower):
    """
    Subclass of FluxTower for data output from EddyPro software.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, flux_bucket, config, filenames=None, **kwargs):
        """

        Parameters
        ----------

        """
        super().__init__(flux_bucket=flux_bucket, config=config, filename=filenames)

        # config

        # data








    # TODO: Figure out raw data import. (Would like to do something similar with get_nearest,
    # but don't really want to mix tower stuff in with uavet.)

    # def get_raw_data(start_time, end_time=None):
        
        # [utils.filename_to_dt(file,end_char=13,dt_format='%Y-%m-%dT%H%M%S',utc_offset=-8) for file in sorted(os.listdir(eri)) if file[0] != '.']



    #     return None


    @staticmethod
    def import_raw(filename,utc_offset=-8):
        """
        Imports raw 10-Hz flux tower data.

        Parameters
        ----------
        filename : str
            The name of the file to be imported

        utc_offset : int, optional
            Fixed UTC offset of the data in hours. The default is -8 (PST).

        Returns
        -------
        raw : DataFrame
            Contains the raw 10-Hz flux data.
        """
        # Import file
        raw = pd.read_csv(filename, skiprows=7, sep='\t', parse_dates={'DateTime':[7,8]})

        # Convert datetime column and add timezone
        raw.DateTime = pd.to_datetime(raw.DateTime, format='%Y-%m-%d %H:%M:%S:%f')

        raw.DateTime = raw.DateTime.apply(utils.make_tzaware,utc_offset=utc_offset)

        # TODO: Fix column names to be universal.
        # TODO: Figure out how to make tz more universally applicable. (Currently specific to JLDP).

        return raw


class CSITower(FluxTower):


    def __init__(self, filename, config):

        super().__init__(filename, config)

        #self.data = self.import_data(filename)









        # # dt_format
        # self.dt_format = config.get('dt_format', '%Y-%m-%dT%H%M%S')

        # # timestamp
        # self.timestamp = utils.filename_to_dt(
        #     self.filename,
        #     skip_char=config.getint('skip_char', 0),
        #     end_char=config.getint('end_char', 13),
        #     dt_format=self.dt_format,
        #     tz_name=self.tz,
        #     utc_offset=-8
        # )
        # # TODO: Figure out UTC offset flexibility. 
        # # TODO: Some of these probs don't need to come from config bc EddyPro output should be the same.
        