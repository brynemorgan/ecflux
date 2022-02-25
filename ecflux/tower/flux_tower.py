import numpy as np
import pandas as pd

import uavet.utils


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
    
    # TODO: Figure out timezone info/pre-processing for tower data.
    df.date_time = df.date_time.apply(uavet.utils.make_tzaware)

    return df



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
        h = h.fillna(1500)

    return h


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
    foot.insert(1,'DateTimeUTC',df.date_time.dt.tz_convert('UTC'))

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
    foot.insert(7,'sigma_v', np.sqrt(df[v_var]))
    foot.drop(v_var,1,inplace=True)

    # Remove rows where u* < 0.1 or h < z
    foot = foot[(foot['u*'] >= 0.1) & (foot.h > z)]

    foot.reset_index(drop=True,inplace=True)

    return foot