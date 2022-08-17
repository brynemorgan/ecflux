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
import sys

# import math
# import random
# import subprocess

# import pytz
import datetime
import numpy as np
import pandas as pd


from scipy import stats


# from uavet import utils
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import rcParams
from mpl_toolkits import mplot3d

%matplotlib qt


# Local imports
import FFP_Python.calc_footprint_FFP as ffp
import FFP_Python.calc_footprint_FFP_climatology as ffp_clim

from tower_utils import import_tower, calc_abl_height, get_ffp_data

import utils


#%%

#%% IMPORT TOWER DATA

# Get home directory + add Box filepath
filepath = os.path.expanduser('~') + '/Box/Dangermond/RamajalTower/'
# filepath = '/Volumes/UAV-ET/Dangermond/RamajalTower/'

# Get list of EddyPro output files with working sonic anemometer (i.e. those 
# after 21 Jan 2021).
files = [txt for txt in sorted(os.listdir(os.path.join(filepath,'summaries'))) if txt[0] != '.']   
# Import flux data.
flux = pd.concat([import_tower(os.path.join(filepath,'summaries',file)) for file in files],ignore_index=True)

# Get IRT data
# irts = pd.read_csv('~/Box/Dangermond/RamajalTower/CR1000X/Data/CR1000X_SN8845_DIR_IRT_OUT.dat',skiprows=[0,2,3], na_values='NAN')
irt_file = os.path.join(filepath, 'CR1000X', 'Data', 'CR1000X_SN8845_DIR_IRT_OUT.txt')
irts = pd.read_csv(irt_file, skiprows=[0,2,3], na_values='NAN')


irts['TIMESTAMP'] = pd.to_datetime(irts.TIMESTAMP)
# irts = irts[irts.DOY > 75]

irts.TIMESTAMP = irts.TIMESTAMP.apply(utils.make_tzaware, tz_name='America/Los_Angeles')

irts.set_index(irts.TIMESTAMP,inplace=True)

irts_hh = irts.resample('30T').mean()


# All data
tower = flux.join(irts_hh,on='date_time',rsuffix='_IRT')

tower.insert(1, 'DATE', tower.date_time.dt.date)

tower['T_s'] = (tower.IRT3_TT_C_Avg+tower.IRT2_TT_C_Avg+tower.IRT3_TT_C_Avg)/3 + 273.15
tower['G'] = (tower.SHF_1_1_1+tower.SHF_2_1_1+tower.SHF_3_1_1)/3
tower['Q_av'] = tower.RN_1_1_1 - tower.G
tower['EBR'] = tower.RN_1_1_1 - tower.H - tower.LE - tower.G
tower['EBR_perc'] = abs(tower.EBR / tower.Q_av)


tower['FLAG'] = 0


tower.FLAG[tower.P_RAIN_1_1_1 != 0] = 11                        # 516
tower.FLAG[(tower.LE.isna()) & (tower.FLAG == 0)] = 12          # 372
tower.FLAG[(tower.SWIN_1_1_1 < -5) & (tower.FLAG == 0)] = 13    # 4380

tower.FLAG[(tower.EBR_perc  > .20) & (tower.FLAG == 0)] = 1     # 8926

# SW + Raining =            146
# SW + LE nan =             229
# Raining + LE nan =        186
# SW + raining + LE nan =    76

del flux
del irts

#%%


# Optional inputs: 
#   wind_dir 
#   rs = [10:10:90]
#   nx (# of grid elements of scaled footprint; i.e. resolution)
#   rslayer


# YYYY, mm, dd, HH_UTC, MM, z_m, d, z0, 'wind_speed', 'L', sigma_v, 'u*', 'wind_dir'

# foot = tower[['date_time','wind_speed', 'L', 'v_var', 'u*', 'wind_dir','daytime']].copy()

# foot.insert(1,'YEAR',foot.date_time.dt.year)
# foot.insert(2,'MONTH',foot.date_time.dt.month)
# foot.insert(3,'DAY',foot.date_time.dt.day)
# foot.insert(4,'HOUR',foot.date_time.dt.hour+8)
# foot.insert(5,'MIN',foot.date_time.dt.minute)

# foot.insert(6,'z_m',3.8735)
# foot.insert(7,'d',0.3*0.67)
# foot.insert(8,'z0',0.3*0.15)

# foot.insert(12,'sigma_v',np.sqrt(foot.v_var))

# foot = foot.drop(['date_time','v_var'],1)

# #foot.to_csv('~/Research/Dangermond/TowerData/foot_data.csv',index=False)

# # daytime
# foot_day = foot[foot.daytime > 0]
# foot_day = foot_day.drop('daytime',1)

# # AM
# foot_am = foot_day[foot_day.HOUR <= 20].copy()

# # PM
# foot_pm = foot_day[foot_day.HOUR > 20].copy()

# # midday
# foot_9_3 = foot_day[(foot_day.HOUR >= 17) & (foot_day.HOUR < 23)].copy()



# foot_day.to_csv('~/Research/Dangermond/TowerData/foot_day.csv',index=False)
# foot_am.to_csv('~/Research/Dangermond/TowerData/foot_am.csv',index=False)
# foot_pm.to_csv('~/Research/Dangermond/TowerData/foot_pm.csv',index=False)
# foot_9_3.to_csv('~/Research/Dangermond/TowerData/foot_93.csv',index=False)



#%% FLIGHTS

flight_times = pd.DatetimeIndex([
    pd.Timestamp('2021-03-01 13:00:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2021-03-24 12:30:00-0700', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-18 10:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-18 11:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-18 12:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-18 13:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-18 14:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-25 10:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-25 11:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-25 12:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-25 13:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-25 14:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-28 12:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-28 13:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-28 14:30:00-0800', tz='America/Los_Angeles'),
    pd.Timestamp('2022-01-28 15:30:00-0800', tz='America/Los_Angeles')
])

flights_tower = tower.loc[tower['date_time'].isin(flight_times)].reset_index(drop=True)


flights_foot = get_ffp_data(flights_tower)

#%% FOR ONLINE FFP 

# flights_csv = pd.DataFrame({
#     'yyyy' : flights_foot.DateTimeUTC.dt.year,
#     'mm' : flights_foot.DateTimeUTC.dt.month,
#     'day' : flights_foot.DateTimeUTC.dt.day,
#     'HH_UTC' : flights_foot.DateTimeUTC.dt.hour,
#     'MM' : flights_foot.DateTimeUTC.dt.minute,
#     'zm' : 3.8735
# })

# flights_csv['d'] = 0.67 * 0.3
# # flights_csv['z0'] = 0.125 * 0.3
# flights_csv['z0'] = -999
# flights_csv[['u_mean', 'L', 'sigma_v', 'u_star', 'wind_dir']] = flights_foot[['wind_speed', 'L', 'sigma_v', 'u*', 'wind_dir']]


# flights_csv.to_csv('/Users/brynmorgan/Research/Dangermond/TowerData/Footprints/FFP_inputs/flights_tower_ffp.csv',index=False)

# out_names = flights_foot.date_time.dt.strftime('%Y-%m-%d-%H-%M')

# folder = '/Users/brynmorgan/Research/Dangermond/TowerData/Footprints/FFP_inputs/'

# for i in range(len(flights_csv)):

#     out_name = 'flights_tower_ffp_' + out_names.loc[i] + '.csv'

#     flights_csv.loc[[i]].to_csv(folder+out_name, index=False)


#%%
rs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


ffp_list = []

for i in range(flights_foot.shape[0]):
    fluxfp = ffp.FFP(
        zm = flights_foot.loc[i].z_m, 
        z0 = None,
        umean = flights_foot.loc[i].wind_speed,
        h = flights_foot.loc[i].h,
        ol = flights_foot.loc[i].L,
        sigmav = flights_foot.loc[i].sigma_v,
        ustar = flights_foot.loc[i]['u*'], 
        wind_dir = flights_foot.loc[i].wind_dir,
        rs = rs,
        nx = 1000
    )
    ffp_list.append(fluxfp)
    print(flights_foot.loc[i].date_time)

clim_list = []

for i in range(flights_foot.shape[0]):
    fluxfp = ffp_clim.FFP_climatology(
        flights_foot.loc[i].z_m, 
        None,
        flights_foot.loc[i].wind_speed,
        flights_foot.loc[i].h,
        flights_foot.loc[i].L,
        flights_foot.loc[i].sigma_v,
        flights_foot.loc[i]['u*'], 
        wind_dir = flights_foot.loc[i].wind_dir,
        dx=1,
        rs=rs,
        crop=1,
        fig=True
    )
    clim_list.append(fluxfp)
    print(flights_foot.loc[i].date_time)



# dy = dx
# if rs is not None:
#     clevs = get_contour_levels(f_2d, dx, dy, rs)
#     frs = [item[2] for item in clevs]
#     xrs = []
#     yrs = []
#     for ix, fr in enumerate(frs):
#         xr,yr = get_contour_vertices(x_2d, y_2d, f_2d, fr)
#         if xr is None:
#             frs[ix] = None
#         xrs.append(xr)
#         yrs.append(yr)

#%%
# tower = pd.concat([flux.import_tower('summaries/'+file) for file in files],ignore_index=True)

# Get footprint data
# foot_all = flux.get_ffp_data(tower)

# foot = foot_all.iloc[-4,:]

# z_m = flights_foot.z_m
# z_0 = None
# umean = flights_foot.wind_speed
# h = flights_foot.h
# L = flights_foot.L
# sigma_v = flights_foot.sigma_v
# u_star = flights_foot['u*']
# wind_dir = flights_foot.wind_dir

# rs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]


# footprint = ffp.FFP(z_m, 
#                     z_0, 
#                     umean, 
#                     h, 
#                     L, 
#                     sigma_v, 
#                     u_star, 
#                     wind_dir,
#                     rs=rs,
#                     dx=1,
#                     crop=1,
#                     fig=True)


#%%
# FFP = myfootprint.FFP(zm,z0,umean,h,ol,sigmav,ustar,optional_inputs)

# foot = flux.get_ffp_data(tower)
foot = get_ffp_data(tower)


foot_day = foot[(foot.date_time.dt.hour > 6) & (foot.date_time.dt.hour < 18)].reset_index(drop=True)
rs = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]
#fluxfp = ffp.FFP(foot.iloc[:,2:-1], axis=1, wind_dir=foot.wind_dir)
ffp_list = []

for i in range(foot.shape[0]):
    fluxfp = ffp.FFP(foot.loc[i].z_m, 
                        None,
                        foot.loc[i].wind_speed,
                        foot.loc[i].h,
                        foot.loc[i].L,
                        foot.loc[i].sigma_v,
                        foot.loc[i]['u*'], wind_dir = foot.loc[i].wind_dir)
    ffp_list.append(fluxfp)
    print(foot.loc[i].date_time)


ffp_list = []

for i in range(foot_day.shape[0]):
    fluxfp = ffp.FFP(foot_day.loc[i].z_m, 
                        None,
                        foot_day.loc[i].wind_speed,
                        foot_day.loc[i].h,
                        foot_day.loc[i].L,
                        foot_day.loc[i].sigma_v,
                        foot_day.loc[i]['u*'], 
                        wind_dir = foot_day.loc[i].wind_dir,
                        fig=0)
    ffp_list.append(fluxfp)
    print(foot_day.loc[i].date_time)


ffp_list = []

for i in [739,1173]:
    fluxfp = ffp.FFP(foot_day.loc[i].z_m, 
                        None,
                        foot_day.loc[i].wind_speed,
                        foot_day.loc[i].h,
                        foot_day.loc[i].L,
                        foot_day.loc[i].sigma_v,
                        foot_day.loc[i]['u*'], 
                        wind_dir = foot_day.loc[i].wind_dir,
                        fig=0)
    ffp_list.append(fluxfp)
    print(foot_day.loc[i].date_time)


ffp31_list = []
fig = plt.figure()
for i,row in enumerate(range(726,foot_day.shape[0])):
    ax = fig.add_subplot(2,3,i+1)
    fluxfp = ffp.FFP(foot_day.loc[row].z_m, 
                        None,
                        foot_day.loc[row].wind_speed,
                        foot_day.loc[row].h,
                        foot_day.loc[row].L,
                        foot_day.loc[row].sigma_v,
                        foot_day.loc[row]['u*'], 
                        wind_dir = foot_day.loc[row].wind_dir,
                        dx=1,
                        rs=rs,
                        crop=1,
                        fig=True)
    ffp31_list.append(fluxfp)
    print(foot_day.loc[row].date_time)


clim31_list = []

for i in range(726,foot_day.shape[0]):
    fluxfp = ffp_clim.FFP_climatology(foot_day.loc[i].z_m, 
                        None,
                        foot_day.loc[i].wind_speed,
                        foot_day.loc[i].h,
                        foot_day.loc[i].L,
                        foot_day.loc[i].sigma_v,
                        foot_day.loc[i]['u*'], 
                        wind_dir = foot_day.loc[i].wind_dir,
                        dx=1,
                        rs=rs,
                        crop=1,
                        fig=True)
    clim31_list.append(fluxfp)
    print(foot_day.loc[i].date_time)




#%%

footprint = ffp_list[3]

clevs = ffp.get_contour_levels(footprint['f_2d'],dx=1,dy=1,rs=rs)

ffp.plot_footprint(footprint['x_2d'],
                    footprint['y_2d'],
                    footprint['f_2d'], clevs=footprint['fr'], show_heatmap=True,iso_labels=True)


def plot_footprint(x_2d, y_2d, fs, clevs=None, show_heatmap=True, normalize=None, 
                   colormap=None, line_width=0.5, iso_labels=None):
    '''Plot footprint function and contours if request'''

    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm
    from matplotlib.colors import LogNorm

    # If input is a list of footprints, don't show footprint but only contours,
    # with different colors
    if isinstance(fs, list):
        show_heatmap = False
    else:
        fs = [fs]

    if colormap is None: colormap = cm.jet
    # Define colors for each contour set
    cs = [colormap(ix) for ix in np.linspace(0, 1, len(fs))]

    # Initialize figure
    fig, ax = plt.subplots(figsize=(12, 10))
    # fig.patch.set_facecolor('none')
    # ax.patch.set_facecolor('none')

    if clevs is not None:
        # Temporary patch for pyplot.contour requiring contours to be in ascending orders
        clevs = sorted(clevs)

        # Eliminate contour levels that were set to None
        # (e.g. because they extend beyond the defined domain)
        clevs = [clev for clev in clevs if clev is not None]

        # Plot contour levels of all passed footprints
        # Plot isopleth
        levs = [clev for clev in clevs]
        for f, c in zip(fs, cs):
            cc = [c]*len(levs)
            if show_heatmap:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = 'w', linewidths=line_width)
            else:
                cp = ax.contour(x_2d, y_2d, f, levs, colors = cc, linewidths=line_width)
            # Isopleth Labels
            if iso_labels is not None:
                pers = [str(int(clev[0]*100))+'%' for clev in clevs]
                fmt = {}
                for l,s in zip(cp.levels, pers):
                    fmt[l] = s
                plt.clabel(cp, cp.levels[:], inline=1, fmt=fmt, fontsize=7)

    # plot footprint heatmap if requested and if only one footprint is passed
    if show_heatmap:
        if normalize == 'log':
            norm = LogNorm()
        else:
            norm = None

        for f in fs:
            pcol = plt.pcolormesh(x_2d, y_2d, f, cmap=colormap, norm=norm)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.gca().set_aspect('equal', 'box')

        cbar = fig.colorbar(pcol, shrink=1.0, format='%.3e')
        #cbar.set_label('Flux contribution', color = 'k')
    plt.show()

    return fig, ax






fig = plt.figure()
levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.contour(footprint['x_2d'],footprint['y_2d'],
                    footprint['f_2d'])
plt.colorbar()


footprint = ffp_list[1]

clevs = ffp.get_contour_levels(footprint['f_2d'],dx=1,dy=1,rs=rs)

ffp.plot_footprint(footprint['x_2d'],
                    footprint['y_2d'],
                    footprint['f_2d'], clevs=clevs, show_heatmap=True,iso_labels=True)


fig = plt.figure()
levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.contour(footprint['x_2d'],footprint['y_2d'],
                    footprint['f_2d'])
plt.colorbar()




footprint = clim31_list[0]


ffp_clim.plot_footprint(footprint['x_2d'],
                    footprint['y_2d'],
                    footprint['fclim_2d'], 
                    clevs=clevs[:3], 
                    show_heatmap=False,iso_labels=True)


fig = plt.figure()
levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

plt.contour(footprint['x_2d'],footprint['y_2d'],
                    footprint['f_2d'])
plt.colorbar()


fig = plt.figure()

plt.contour(footprint['x_2d'],footprint['y_2d'],footprint['fclim_2d'],levels=c_lev[-3:])
plt.colorbar()



# FFP_clim output
# FFP      = Structure array with footprint climatology data for measurement at [0 0 zm] m
# x_2d	    = x-grid of 2-dimensional footprint [m]
# y_2d	    = y-grid of 2-dimensional footprint [m]
# fclim_2d = Normalised footprint function values of footprint climatology [m-2]
# rs       = Percentage of footprint as in input, if provided
# fr       = Footprint value at r, if r is provided
# xr       = x-array for contour line of r, if r is provided
# yr       = y-array for contour line of r, if r is provided
# n        = Number of footprints calculated and included in footprint climatology
# flag_err = 0 if no error, 1 in case of error, 2 if not all contour plots (rs%) within specified domain,
#             3 if single data points had to be removed (outside validity)



# FFP output
# x_ci_max = x location of footprint peak (distance from measurement) [m]
# x_ci	 = x array of crosswind integrated footprint [m]
# f_ci	 = array with footprint function values of crosswind integrated footprint [m-1] 
# x_2d	 = x-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
# y_2d	 = y-grid of 2-dimensional footprint [m], rotated if wind_dir is provided
# f_2d	 = footprint function values of 2-dimensional footprint [m-2]
# rs       = percentage of footprint as in input, if provided
# fr       = footprint value at r, if r is provided
# xr       = x-array for contour line of r, if r is provided
# yr       = y-array for contour line of r, if r is provided
# flag_err = 0 if no error, 1 in case of error
