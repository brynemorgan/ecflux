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

import math
import random
import subprocess

import pytz
import datetime
import numpy as np
import pandas as pd


from scipy import stats



import uavet
from uavet.buckets import FlirImageBucket
from uavet.config import ProductionConfig 
from uavet import utils
from uavet import Flight
from uavet import FlirImage
from uavet.images import FlirImageFolder
from uavet import et
from uavet import met_utils as met
from uavet import temp_utils as temp
from uavet import most

from timer import Timer


import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.dates as mdates
from matplotlib import rcParams
from mpl_toolkits import mplot3d


# %matplotlib qt


%matplotlib qt



sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)),'FFP_Python'))

import calc_footprint_FFP as ffp
import calc_footprint_FFP_climatology as ffp_clim
import flux_tower as flux


#%% FUNCTIONS

def import_tower(file, skiprows=[1], parse_dates=[[2,3]], delim_whitespace=True, **kwargs):
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
    
    TO DO:
    -----
        - May want to skip some of the unnecessary columns (esp. at the beginning).
        - Could also set datetime index.
        - May want to create a .txt file with column names to separately import.
    """

    df = pd.read_csv(file, skiprows=skiprows, parse_dates=parse_dates, 
                     delim_whitespace=delim_whitespace, **kwargs)

    df.date_time = df.date_time.apply(utils.make_tzaware, utc_offset=-8, tz_name='America/Los_Angeles')

    return df




#%%



irts = pd.read_csv('~/Box/Dangermond/RamajalTower/CR1000X/Data/CR1000X_SN8845_DIR_IRT_OUT.dat',skiprows=[0,2,3], na_values='NAN')


irts['TIMESTAMP'] = pd.to_datetime(irts.TIMESTAMP)
irts = irts[irts.DOY > 75]

irts.TIMESTAMP = irts.TIMESTAMP.apply(utils.make_tzaware, tz_name='America/Los_Angeles')

irts.set_index(irts.TIMESTAMP,inplace=True)

irts_hh = irts.resample('30T').mean()





#%%

# Average



#%% MAIN
def main():
    
#%% CHANGE DIRECTORY

    # Get home directory + add Box filepath
    filepath = os.path.expanduser('~') + '/Box/Dangermond/RamajalTower/'
    # Change directory 
    os.chdir(filepath)

#%% CONNECT TO NETWORK DIRECTORY

    # Mount smb
    #os.system("osascript -e 'mount volume \"smb://smb.eri.ucsb.edu/brynmorgan\"'")

    # Mount smb
    # os.system("osascript -e 'mount volume \"smb://smb.eri.ucsb.edu\"'")

    # A window will pop up asking which volume you want to mount. Select your home
    # folder (e.g. brynmorgan).

    # If your login info isn't saved in your machine, you can specify a username
    # and password as follows:
    # os.system("osascript -e 'mount volume \"smb://server/servershare\" as user name \"myUserName\" with password \"myPassword\"'")
    
    # Get home folder
    home = os.listdir('/Volumes/')[0]   # This is where mine is; may be different if other things are mounted or name is different.

    # With smb mounted, navigate to the waves-dangermond directory ('/Volumes/home/waves-dangermond/UAV/').
    filepath = os.path.join('/Volumes',home,'waves-dangermond','RamajalTower','')

    # Change directory
    os.chdir(filepath)

#%% DATA IMPORT

    # The summaries folder contains the EddyPro output files, described here: 
    # https://www.licor.com/env/support/EddyPro/topics/output-files-full-output.html
    # NOTE: The timestamps correspond to the END of the averaging period.

    # Get list of EddyPro output files with working sonic anemometer (i.e. those 
    # after 21 Jan 2021).
    files = [tab for tab in sorted(os.listdir(os.path.join(filepath,'summaries'))) if tab[0] != '.']   
    # Import flux data.
    tower = pd.concat([import_tower('summaries/'+file) for file in files],ignore_index=True)



#%%


    def get_ffp_data(
            df, 
            z = 3.8735, 
            d = None,
            z_0 = None, 
            dt = 'date_time',
            u = 'wind_speed', 
            L = 'L', 
            v_var = 'v_var', 
            u_star = 'u*', 
            wind_dir = 'wind_dir',
            lat =  34.526745,
            long = -120.415905,
        ):
        """
        Extracts the data required for footprint calculation from a DataFrame.

        Parameters
        ----------
        df : DataFrame
            The DataFrame with the 30-min flux data.
        
        z : float
            Measurement height [m]. The default is 3.8735.
        
        d : float
            Displacement height [m]. The default is 0.67*z.
        
        z_0 : float
            Roughness length [m]. The default is 0.15*z.
        
        dt : str
            The name of the column of df containing the timestamp of the record.
            The default is 'date_time'.
        
        u : str
            The name of the column of df containing the mean wind speed data in
            m s-1. The default is 'wind_speed'.
        
        L : str
            The name of the column of df containing the Obukhov length in m. The
            default is 'L'.
        
        v_var : str
            The name of the column of df containing the variance of the lateral
            velocity fluctuations in m2 s-2. The default is 'v_var'.
        
        u_star : str
            The name of the column of df containing the friction velocity data in
            m s-1. The default is 'u*'.
        
        wind_dir : str
            The name of the column of df containing the mean wind direction in deg.
            The default is 'wind_dir'.
        
        lat : float
            The latitude of the tower in decimal degrees.
        
        long : float
            The longitude of the tower in decimal degrees.


        Returns
        -------
        foot : DataFrame
            DataFrame containing the variables required to calculate the flux
            footprint.
        """

        # Variables required for ffp.FFP function: zm, z0, u, h, L, sigma_v, u*, wind_dir (optional)

        # Get desired variables
        foot = df[[dt,u,L,v_var,u_star,wind_dir]].copy()

        # Insert UTC time
        foot.insert(1,'DateTimeUTC',df.date_time.dt.tz_localize('US/Pacific').dt.tz_convert('UTC'))

        # Height above displacement height, z_m [m]
        if d is None:
            d = 0.67 * z
        foot.insert(2, 'z_m', z-d)
        # Roughness length, z_0 [m]
        if z_0 is None:
            z_0 = 0.15 * z
        foot.insert(3, 'z_0', z_0)
        # Height of the ABL
        h = calc_abl_height(df[L],df[u_star],lat)
        foot.insert(5, 'h', h)

        # Standard deviation of lateral wind velocity fluctuation [m s-1]
        foot.insert(7,'sigma_v', np.sqrt(v_var))
        foot.drop(v_var,1,inplace=True).reset_index(drop=True)

        return foot








    def calc_abl_height(L,u_star,lat):
        """
        Calculates the height of the boundary layer.

        Parameters
        ----------
        L : float or array-like
            Obukhov length [m]
            Length must match length of u_star.

        u_star : float or array-like
            Friction velocity [m s-1]
            Length must match length of L.

        lat : float
            Latitude in decimal degrees.

        Returns
        -------
        h : float or array-like
            Height of the ABL [m].
        """
        omega = 7.2921e-5   # angular velocity of the earth's rotation [rad s-1]
        
        # For convective conditions, set h = 1500
        if isinstance(L,float):
            if L < 0:
                h = 1500

        # Calculate coriolis parameter [s-1] (https://en.wikipedia.org/wiki/Coriolis_frequency)
        f = 2 * omega * np.sin(np.radians(lat))

        # Calculate h
        h = (L / 3.8) * (-1 + np.sqrt(1 + 2.28 * (u_star / (f * L) )))

        # For convective conditions, set h = 1500
        if isinstance(h,pd.core.series.Series):
            h.fillna(1500)

        return h









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

    footprint = ffp31_list[1]

    clevs = ffp.get_contour_levels(footprint['f_2d'],dx=1,dy=1,rs=rs)

    ffp.plot_footprint(footprint['x_2d'],
                       footprint['y_2d'],
                       footprint['f_2d'], clevs=footprint['fr'], show_heatmap=True,iso_labels=True)


    fig = plt.figure()
    levels = [0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

    plt.contour(footprint['x_2d'],footprint['y_2d'],
                       footprint['f_2d'])
    plt.colorbar()


    footprint = ffp_list[1]

    clevs = ffp.get_contour_levels(footprint['f_2d'],dx=1,dy=1,rs=rs)

    ffp.plot_footprint(footprint['x_2d'],
                       footprint['y_2d'],
                       footprint['f_2d'], clevs=footprint['fr'], show_heatmap=True,iso_labels=True)


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



#%%
if __name__ == "__main__":
    main()