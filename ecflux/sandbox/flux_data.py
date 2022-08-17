#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Fri 21 May 21 15:28:17'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           flux_data.py
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
    'raw_folder': '~/Box/Dangermond/RamajalTower/raw_unzipped/data/',
    'biomet_folder': '~/Box/Dangermond/RamajalTower/raw_unzipped/biomet/',
    'dt_format': '%Y-%m-%dT%H%M%S',
    'utc_offset' : -8,
    'tz': 'America/Los_Angeles',
    'skip_char': 0,
    'end_char': 13
}


#%%

class FluxData():
    """

    30-min averaged flux data from an eddy covariance tower.

    """

    def __init__(self, filename, config=None, **kwargs):

        # filename
        self.filename = filename

        # config
        if config:
            self.config = config
            # tz
            self.tz = self.config.get('tz','America/Los_Angeles')
            # utc_offset
            self.utc_offset = self.config.getint('utc_offset',-8)
        
        # data
        self.data = None


class EddyProFluxData(FluxData):
    """
    Subclass of FluxData for data output from EddyPro software.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, filename, config, **kwargs):
        """

        Parameters
        ----------

        """
        super().__init__(filename, config)

        # timestamp
        self.timestamp = utils.filename_to_dt(
            self.filename,
            skip_char=self.config.getint('skip_char', 0),
            end_char=self.config.getint('end_char', 13),
            dt_format=self.config.get('dt_format', '%Y-%m-%dT%H%M%S'),
            tz_name=self.tz,
            utc_offset=self.utc_offset
        )
        # TODO: Some of these probs don't need to come from config bc EddyPro output should always be the same.


    def init(self):
        """
        Initialize the EddyProFluxData object by importing the data.

        Yields
        ------
        self.data

        """
        # Import data
        self.data = self.import_data(
                self.filename, 
                utc_offset=self.utc_offset,
                tz=self.tz
            )

    def get_data(self):
        """
        Gets the flux data.

        Returns
        -------
        self.data : DataFrame
            Contains the data from the file.
        """
        if self.data is None:
            self.init()
        return self.data

    @staticmethod
    def import_data(filename, skiprows=[1], parse_dates=[[2,3]], delim_whitespace=True, utc_offset=-8, tz='America/Los_Angeles', **kwargs):
        """
        Import 30-min-averaged flux data from EddyPro output file.

        Parameters
        ----------
        file : str
            The name of the file to be imported.
        
        skiprows : list-like, int
            Rows of table to skip; passed to pd.read_csv(). The default is [1] 
            (the row with the units).
        
        parse_dates : bool, list of ints, or list of lists
            Columns of file to parse into datetime objects; passed to pd.read_csv().
            The default is [[2,3]], which parses the date and time columns into a 
            single datetime column. This should not need to be changed.
        
        delim_whitespace : bool
            Specifies whether or not to interpret whitespace as the delimiter between
            columns; passed to pd.read_csv(). Equivalent to setting sep='\s+'. This 
            parameter is magical. The default is True.
        
        utc_offset : int
            The UTC offset of the dataset in hours (passed to uavet.utils.make_tzaware). 
            The default is -8.
        
        tz : str
            Name of timezone. Must be one of the valid timezones listed in pytz.alltimezones.
            The default is 'America/Los_Angeles'. If using utc_offset, tz_name should
            correspond to the desired output timezone.

        **kwargs 
            kwargs to be passed to pd.read_csv(). Options can be found in the documentation
            (https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)
        
        Returns
        -------
        df : DataFrame
            Contains data imported from file.
        
        # TODO:
        -----
            - May want to skip some of the unnecessary columns (esp. at the beginning).
            - Could also set datetime index.
            - May want to create a .txt file with column names to separately import.
        """
        # Import flux data
        df = pd.read_csv(filename, skiprows=skiprows, parse_dates=parse_dates, 
                        delim_whitespace=delim_whitespace, **kwargs)
        # Assign timezone
        df.date_time = df.date_time.apply(utils.make_tzaware, utc_offset=utc_offset, tz_name=tz)

        return df




class CSIFluxData(FluxData):
    """
    Subclass of FluxData for data output from a Campbell datalogger.

    Attributes
    ----------

    Methods
    -------

    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def init(self):
        """
        Initializes the CSIFluxData object by importing the data.

        Yields
        ------
        self.data

        """
        # Import data
        self.data = self.import_data(
                self.filename, 
                #tz=self.tz     Figure out how to implement without changing config
            )

    def get_data(self):
        """
        Gets the flux data.

        Returns
        -------
        self.data : DataFrame
            Contains the data from the file.
        """
        if self.data is None:
            self.init()
        return self.data


    @staticmethod
    def import_data(filename, tz='Africa/Nairobi', **kwargs):

        # Import data
        df = pd.read_csv(
            filename, 
            skiprows=[0,2,3], 
            delim_whitespace=False, 
            na_values="NAN", 
            header=0,
            parse_dates=[0],
            **kwargs
        )

        # Assign timezone
        df.TIMESTAMP = df.TIMESTAMP.apply(utils.make_tzaware, tz_name=tz)

        return df


# TODO: Add FluxNetData class.
#   - Read metadata file to retrieve things like canopy height, utc offset, etc.



