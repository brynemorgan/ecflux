#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Wed 03 Mar 21 17:29:25'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           tower_footprint.py
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
import math
import datetime

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import rcParams
from mpl_toolkits import mplot3d


%matplotlib qt


sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'FFP_Python'))

import calc_footprint_FFP as ffp
import calc_footprint_FFP_climatology as ffp_clim
import flux_tower as flux


#%% FUNCTIONS



#%% MAIN
def main():

#%% DATA IMPORT

    # Get home directory + add Box filepath
    filepath = os.path.expanduser('~') + '/Box/Dangermond/Ramajal Tower/'
    # Change directory 
    os.chdir(filepath)

    # The summaries folder contains the EddyPro output files, described here: 
    # https://www.licor.com/env/support/EddyPro/topics/output-files-full-output.html
    # NOTE: The timestamps correspond to the END of the averaging period.

    # Get list of EddyPro output files with working sonic anemometer (i.e. those 
    # after 21 Jan 2021).
    files = [tab for tab in os.listdir(os.path.join(filepath,'summaries'))][379:]     
    # Import flux data.
    tower = pd.concat([flux.import_tower('summaries/'+file) for file in files],ignore_index=True)

    # Get footprint data
    foot_all = flux.get_ffp_data(tower)

    foot = foot_all.iloc[-4,:]

    z_m = foot.z_m
    z_0 = None
    umean = foot.wind_speed
    h = foot.h
    L = foot.L
    sigma_v = foot.sigma_v
    u_star = foot['u*']
    wind_dir = foot.wind_dir

    rs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


    footprint = ffp.FFP(z_m, 
                        z_0, 
                        umean, 
                        h, 
                        L, 
                        sigma_v, 
                        u_star, 
                        wind_dir,
                        rs=rs,
                        dx=1,
                        crop=1,
                        fig=True)



#%%
if __name__ == "__main__":
    main()