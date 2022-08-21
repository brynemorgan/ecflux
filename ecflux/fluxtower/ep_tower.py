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
from fluxtower.utils import get_recols,import_dat

# FUNCTIONS

class EddyProTower(FluxTower):

    def __init__(self, filepath, meta_file, biomet_files=None):

        super().__init__(filepath, meta_file, biomet_files)

        # _flux_files
        self._flux_files = self.get_flux_files()

        # flux
        self.flux = pd.concat(
            [self.import_flux(
                os.path.join(self._filepath,'summaries',file),index_col=0
            ) for file in self._flux_files]
        )

        # data
        self.set_data()
    
    def get_flux_files(self):

        files = [ txt for txt in sorted(
            os.listdir(os.path.join(self._filepath, 'summaries'))
        ) if txt[0] != '.' ]

        return files


    @staticmethod
    def import_flux(file, skiprows=[1], parse_dates=[[2,3]], delim_whitespace=True, **kwargs):
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

