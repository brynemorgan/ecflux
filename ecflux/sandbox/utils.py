#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2022'

__license__ = 'MIT'
__date__ = 'Thu 07 Jul 22 13:04:08'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           utils.py
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


import datetime
import pytz
import timezonefinder as tzf


# FUNCTIONS

def make_tzaware(dt_naive, tz_name='America/Los_Angeles', coords=None, utc_offset=None):
    """
    Converts naive datetime object to timezone-aware object based on one of the
    following:
        1. Timezone name (default)
        2. Geographic coordinates
        3. Fixed UTC offset

    Note: the timestamp provided is not *converted* to the given timezone; the
    timezone is simply attached. Thus the provided timezone info should be that
    of the naive datetime object.

    Parameters
    ----------
    dt_naive : datetime object
        Naive datetime object to be made aware.

    tz_name : str
        Name of timezone. Must be one of the valid timezones listed in pytz.alltimezones.
        The default is 'America/Los_Angeles'. If using utc_offset, tz_name should
        correspond to the desired output timezone.

    coords : tuple
        Tuple object containing the (lat,long) from which to identify a timezone.
        The default is None.

    utc_offset : int
        Fixed UTC offset in hours.


    Returns
    -------
    dt : datetime object
        Timezone-aware datetime object.

    """

    # 1. Coordinates
    #       Note: only do this if timestamp is DST-aware, don't use dt.replace(tzinfo=tz_name)
    if isinstance(coords, tuple):
        tz = pytz.timezone(tzf.TimezoneFinder().timezone_at(
            lng=coords[1], lat=coords[0]))
        dt = tz.localize(dt_naive)
    # 2. Fixed UTC offset
    elif utc_offset:
        tz = datetime.timezone(datetime.timedelta(hours=utc_offset))
        dt = dt_naive.replace(tzinfo=tz)
        # Convert to timezone
        dt = dt.astimezone(pytz.timezone(tz_name))
    # 3. Timezone name
    #       Note: only do this if timestamp is DST-aware, don't use dt.replace(tzinfo=tz_name)
    else:
        # Check if name is valid timezone
        if tz_name in pytz.all_timezones:
            # Add timezone
            tz = pytz.timezone(tz_name)
            dt = tz.localize(dt_naive)
        else:
            raise ValueError(
                'The timezone provided is not valid. To see a list of valid timezones, run pytz.alltimezones.')

    return dt