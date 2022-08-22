#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Sat 20 Aug 22 00:27:36'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           csi_tower.py
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

# SUB-CLASS

class CSITower(FluxTower):

    def __init__(self, filepath, meta_file, flux_file, biomet_files=None):

        super().__init__(filepath, meta_file, biomet_files)

        # _flux_file
        self._flux_file = flux_file

        # flux
        self.flux = self.import_flux()

        # # biomet
        # self.biomet = self.import_biomet()
        # data
        self.set_data()
        self.clean_data()


    def import_flux(self):

        return import_dat(self._flux_file)        


    # def import_biomet(self):

    #     biomet_dfs = [ import_dat(os.path.realpath(file)) for file in self._biomet_files ]

    #     # biomet = rad.join(soil, how='inner', rsuffix='_Table1')
    #     return biomet_dfs