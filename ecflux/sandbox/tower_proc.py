#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Thu 11 Feb 21 11:33:41'
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


#%% FIGURE PARAMS


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
mpl.rcParams['font.size'] = 10.0


mpl.rcParams['font.family'] = 'sans-serif'
mpl.rcParams['font.sans-serif'] = 'Myriad Pro'
mpl.rcParams['font.size'] = 9.0



# Figure Params
# mpl.rcParams['font.family'] = 'sans-serif'
# mpl.rcParams['font.sans-serif'] = 'Verdana'
# mpl.rcParams['font.size'] = 9.0

gray = '#454545'

# turq = '#409099'
turq = '#2195AC'
sky = '#95C7CB'
salmon = '#f9ab98'
rouge = '#EF877D'
forest = '#1E6626'
# green = '#B2CE5B'
green = '#81BF24'
gold = '#F1C300'
brown = '#AA7942'

maroon = '#80004D'
lav = '#A6A6ED'

c_list = [gold, turq, rouge, green]
c3_list = [green,gray,turq]
c4_list = [gray,rouge,turq,green]

co_list = [green, rouge, turq, gray, gold, sky]
    
#col_list = [sky, gold, lav, maroon, gray, rouge, forest]

col_list = [green, rouge, turq, gray, sky, gold, forest, lav, maroon]

e_list = ['#6a7b36','#e74839','#295d63','k','#62acb2','#a58500','#0d2b10','#6666e0','#34001f']

red = '#cc3928'


#%%

# Set datetime index
tower.set_index(tower.date_time,inplace=True)

# Resample + fill missing data with NaNs
ts = tower.resample('30T').asfreq()

ts['week'] = ts.index.isocalendar()['week']

ts['co2_gm2'] = ts.co2_flux * 1e-6 * 44.01 * 60 * 30



#%%


g = ts[['SHF_1_1_1','SHF_2_1_1','SHF_3_1_1']].mean(axis=1)

#g1 = tower[['SHFSENS_1_1_1','SHFSENS_2_1_1','SHFSENS_3_1_1']].mean(axis=1)

Q_av = ts.RN_1_1_1+g 
resid = ts.RN_1_1_1 - g - ts.H - ts.LE

#%%

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

# '#680082'
ax1.plot(ts.index[ts.RN_1_1_1 > -200], ts.RN_1_1_1[ts.RN_1_1_1 > -200], color=maroon, alpha = 0.8, label=r'$R_n$')
ax1.plot(ts.index, ts.H, color=green, alpha=0.8, label=r'$H$')
ax1.plot(ts.index, ts.LE, color=turq, alpha=0.8, label=r'$\lambda E$')
ax1.plot(ts.index, g, color='#FF7A00', alpha=0.8, label=r'$G$')
#ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')


ax1.set_ylabel(r"Flux (W m$^{-2}$)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()

#%%


fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

# '#680082'
ax1.plot(ts.groupby(ts.index.isocalendar()['week']).RN_1_1_1.max(), color=maroon, alpha = 0.8, label=r'$R_n$')
ax1.plot(ts.groupby(ts.week).H.max(), color=green, alpha=0.8, label=r'$H$')
ax1.plot(ts.groupby(ts.week).LE.max(), color=turq, alpha=0.8, label=r'$\lambda E$')
ax1.plot(g.groupby(ts.week).max(), color='#FF7A00', alpha=0.8, label=r'$G$')
# ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')

ax1.set_ylabel(r"Flux (W m$^{-2}$)")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
# plt.gcf().autofmt_xdate()
plt.legend()



fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

# '#680082'
ax1.plot(ts.groupby(ts.week).co2_flux.mean(), color=green, alpha=0.8, label=r'CO$_2$ flux')
ax1.plot(ts.groupby(ts.week).h2o_flux.mean(), color=turq, alpha=0.8, label=r'H$_2$O flux')
# ax1.plot(g.groupby(ts.week).max(), color='#FF7A00', alpha=0.8, label=r'$G$')
# ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')

ax1.set_ylabel(r"Flux (W m$^{-2}$)")
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
# plt.gcf().autofmt_xdate()
plt.legend()



#%% DAILY/WEEKLY H2O + CO2 FLUXES

def date_from_wk(year, week):
    first = datetime.date(year, 1, 1)
    base = 1 if first.isocalendar()[1] == 1 else 8
    return first + datetime.timedelta(days=base - first.isocalendar()[2] + 7 * (week - 1))

wks = [date_from_wk(2021,wk) for wk in range(3,27)]


et_mmday = ts.groupby(ts.index.date).ET.sum(min_count=24)*0.5
et_mmday.index = pd.to_datetime(et_mmday.index)

co2_gm2day = ts.groupby(ts.index.date).co2_gm2.sum(min_count=24)
co2_gm2day.index = pd.to_datetime(co2_gm2day.index)


# PLOTS
fig = plt.figure(figsize=(6.0,6.0))
ax1 = fig.add_subplot(3,1,1)
ax1.plot(et_mmday, color=green, alpha=0.45, label=r'daily')
# ax1.plot(wks,et_mmday.groupby(et_mmday.index.isocalendar()['week']).mean(), color='b', alpha=0.8, label=r'weekly average')
ax1.plot(et_mmday.rolling(7, min_periods=1, center=True).mean(), color='green', alpha=0.8, label=r'7-day average')

ax1.set_ylabel(r"Evapotranspiration (mm day$^{-1}$)")
ax1.set_yticks([0.0,1.0,2.0,3.0])
# ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
# plt.gcf().autofmt_xdate()
ax1.annotate('(a)', xy=(0.01,0.902), xycoords='axes fraction')

plt.legend(loc='upper right')

ax2 = fig.add_subplot(3,1,2)
ax2.plot(co2_gm2day[abs(co2_gm2day) <= 25.0], color='white', alpha=0.8)
ax2.axhline(color='k',alpha=0.8,linestyle='--',linewidth=0.9)
ax2.plot(co2_gm2day[abs(co2_gm2day) <= 25.0], color=lav, alpha=0.45, label=r'daily')
# ax2.plot(wks,co2_gm2day.groupby(co2_gm2day.index.isocalendar()['week']).mean(), color='g', alpha=0.8, label=r'weekly average')
ax2.plot(co2_gm2day[abs(co2_gm2day) <= 25.0].rolling(7, min_periods=1, center=True).mean(), color=maroon, alpha=0.8, label=r'7-day average')
# ax2.axhline(color='k',alpha=0.8,linestyle='--',linewidth=0.9)

# ax2.set_ylabel(r"CO$_2$ flux (g m$^{-2}$ day$^{-1}$)")
# ax2.set_ylabel(r"Net ecosystem exchange (g m$^{-2}$ day$^{-1}$)")
ax2.set_ylabel(r"NEE (g m$^{-2}$ day$^{-1}$)")
# ax2.set_ylim([-28,28])
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
plt.gcf().autofmt_xdate()

ax2.annotate('(b)', xy=(0.01,0.9), xycoords='axes fraction')

plt.legend(loc='upper right')


ax3 = fig.add_subplot(3,1,3)
ax3.bar(np.unique(ts.index.date),ts.groupby(ts.index.date).P_RAIN_1_1_1.sum()*-1000, color=turq, alpha=0.6)
ax3.set_ylabel('Precipitation (mm)')
# ax3.set_yticks([0.0,-0.03,-0.06,-0.09,-0.12])
# ax3.set_yticklabels([0.0,0.03,0.06,0.09,0.12])
ax3.set_yticks([0.0,-30.,-60.,-90.,-120.])
ax3.set_yticklabels([0,30,60,90,120])

ax3.annotate('(c)', xy=(0.01,0.9), xycoords='axes fraction')

ax4 = ax3.twinx()

ax4.plot(ts.index, ts.SWC_1_1_1, label='2.5"', color='orange', alpha=0.8)
ax4.plot(ts.index, ts.SWC_2_1_1, label='9.5"', color=red, alpha=0.8)
ax4.set_ylabel('Soil water content (m' + r"$^3$" + ' m' + r"$^{-3}$"+')')
ax4.set_yticks([0.0,0.1,0.2,0.3,0.4])
ax4.legend(loc='upper right')

plt.tight_layout()

plt.savefig('/Users/brynmorgan/Research/Dangermond/Figures/Tower_trans1.png',dpi=300,transparent=True)









fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(co2_gm2day, color=green, alpha=0.8, label=r'CO$_2$')
ax1.set_ylabel(r"CO$_2$ flux (g m$^{-2}$ day$^{-1}$)")

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
plt.gcf().autofmt_xdate()


fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(et_mmday, color=turq, alpha=0.8, label=r'$E$')
ax1.plot(et_mmday.groupby(et_mmday.index.isocalendar()['week']).mean(), color=turq, alpha=0.8, label=r'$E$')
ax1.set_ylabel(r"Evapotranspiration (mm day$^{-1}$)")

ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %Y'))   
plt.gcf().autofmt_xdate()




fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
ax1.plot(ts.qc_co2_flux)
ax1.plot(ts.co2_flux)






#%%

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.plot(ts.index, ts.RN_1_1_1, color=maroon, alpha = 0.6, label=r'$R_n$')
ax1.plot(ts.index, ts.H, color=turq, alpha=0.8, label=r'$H$')
ax1.plot(ts.index, ts.LE, color=green, alpha=0.8, label=r'$\lambda E$')
ax1.plot(ts.index, g, color=red, alpha=0.8, label=r'$G$')
ax1.plot(ts.index, Q_av, color=salmon, alpha=0.8, label=r'$Q_{av}$')
ax1.plot(ts.index, resid, color=gray, alpha=0.8, label=r'$Res$')
#ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')


ax1.set_ylabel(r"Flux (W m$^{-2}$)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.plot(ts.index, (resid/Q_av)*100, color=gray, alpha=0.8, label='Residual')
#ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')


ax1.set_ylabel(r"Residual (%)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()


#%%

fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.plot(ts.groupby(ts.index.date).min().index,ts.LE.groupby(ts.index.date).max())
ax1.set_ylabel(r"Flux (W m$^{-2}$)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()



fig = plt.figure()

ax1 = fig.add_subplot(2,1,1)

ax1.plot(ts.index, ts.H+ts.LE+g, color=maroon, alpha = 0.6, label=r'$R_n$')
ax1.plot(ts.index, ts.H, color=turq, alpha=0.8, label=r'$H$')
ax1.plot(ts.index, ts.LE, color=green, alpha=0.8, label=r'$\lambda E$')
ax1.plot(ts.index, g, color=red, alpha=0.8, label=r'$G$')
ax1.plot(ts.index, ts.RN_1_1_1-(ts.H+ts.LE+g), color=gray, alpha = 0.6, label='residual')


ax1.set_ylabel(r"Flux (W m$^{-2}$)")
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()

fig = plt.figure()
ax2 = fig.add_subplot(1,1,1)

ax2.plot(ts.index, ts.RN_1_1_1,'.', color=maroon, alpha = 0.6, label=r'$R_n$')
ax2.plot(ts.index, ts.SWIN_1_1_1, '.', color=forest, alpha=0.8, label=r'$R^{\downarrow}_{_{SW}}$')
ax2.plot(ts.index, ts.SWOUT_1_1_1, '.', color=green, alpha=0.8, label=r'$R^{\uparrow}_{_{SW}}$')
ax2.plot(ts.index, ts.LWIN_1_1_1, '.', color=red, alpha=0.8, label=r'$R^{\downarrow}_{_{LW}}$')
ax2.plot(ts.index, ts.LWOUT_1_1_1, '.', color=salmon, alpha=0.8, label=r'$R^{\uparrow}_{_{LW}}$')

#ax2.plot(ts.index, g, color=red, alpha=0.8, label=r'$G$')
ax2.set_ylabel(r"Radiation (W m$^{-2}$)")
ax2.legend(bbox_to_anchor=(1.02,1), loc="upper left")



#ax1.xaxis.set_minor_locator(mdates.HourLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b %H:%M'))   
plt.gcf().autofmt_xdate()
plt.legend()



# Prevailing wind direction

np.sin(np.radians(ts.wind_dir)).sum()
np.cos(np.radians(ts.wind_dir)).sum()

wd = (180/np.pi) * np.arctan2(np.sin(np.radians(ts.wind_dir)).sum(), np.cos(np.radians(ts.wind_dir)).sum())




fig = plt.figure()

ax1 = fig.add_subplot(3,1,1)

ax1.plot(ts.index, ts.wind_speed*2.25, color=gray, alpha=0.8)
ax1.set_ylabel(r"Wind speed (m s$^{-1}$)")

ax1.annotate('Prevailing wind direction: ' + '{0:.1f}'.format(wd) + r'$^{\circ}$ NE', xy=(0.5,0.9),xycoords='axes fraction')

ax3 = fig.add_subplot(3,1,2)

ax3.plot(ts.index, ts.max_wind_speed*2.25, color=turq, alpha=0.8)
ax3.set_ylabel(r"Max wind speed (m s$^{-1}$)")

ax2 = fig.add_subplot(3,1,3)

ax2.plot(ts.index, ts.air_temperature-273.15, color=maroon, alpha=0.8)
ax2.set_ylabel(r"Temperature ($^{\circ}$C)")


#ax1.xaxis.set_minor_locator(mdates.HourLocator())
ax2.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))   
plt.gcf().autofmt_xdate()



fig = plt.figure()

ax1 = fig.add_subplot(1,1,1)

ax1.plot(ts.VPD/1000,ts.ET, '.', color=turq, alpha=0.8)

ax1.set_xlabel('VPD (kPa)')
ax1.set_ylabel(r'ET (mm hr$^{-1})$')


#%%


fig = plt.figure()

plt.plot(ts.index, ts.SWC_1_1_1, label='2.5 in')
plt.plot(ts.index, ts.SWC_2_1_1, label='9.5 in')
plt.ylabel('Volumetric water content (m' + r"$^3$" + ' m' + r"$^{-3}$"+')')
plt.legend()


#%%

fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
plt.plot(ts.index, ts.SWIN_1_1_1)
ax1.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))   
plt.gcf().autofmt_xdate()



