#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Fri 21 May 21 12:49:25'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           buckets.py
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
import pandas as pd

from uavet.buckets import Bucket
from .tower import EddyProTower, FluxTower
from .flux_data import EddyProFluxData, FluxData

#%%

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

# Flux Tower Data
# [flux]
# dir: ~/Box/Dangermond/RamajalTower
# raw_dir: ${dir}/raw_unzipped/data
# biomet_dir: ${dir}/raw_unzipped/data
# dt_format: %Y-%m-%dT%H%M%S
# utc_offset: -8
# tz: America/Los_Angeles
# skip_char: 0
# end_char: 13

#%% 

class FluxBucket(Bucket):
    """

    A bucket of flux data.

    """
    def __init__(self, bucket_object, config):
        super().__init__(bucket_object=bucket_object, config=config)


    def get_objects(self):
        
        # Initialize empty list
        objects = []
        for file in [f for f in sorted(os.listdir(self.dir)) if f[0] != '.']:
            # Get full path of each file in bucket
            full_path = os.path.join(self.dir, file)
            # Check that path is a file
            if os.path.isfile(full_path):
                objects.append(self.object_type(full_path, self.config))

        return objects

    def get_all_data(self):
        """
        Return the entire contents of the bucket as a single object.

        Returns
        -------
        all_obj : EddyProFluxData
            All flux data as a single object.
        
        """
        all_obj = self.merge_data(self.objects)

        return all_obj

    def get_nearest(self, start_time, end_time=None):
        """
        Get the objects in the bucket corresponding to a timestamp or range of
        timestamps and combine into a single object.

        Parameters
        ----------
        start_time : datetime
            The datetime that is being matched to bucket objects.

        end_time : datetime
            The ending datetime that is being matched to bucket objects. A list 
            containing all matching objects will be returned. The default is none.

        Returns
        -------
        merged_obj : flux_data.FluxData
            Contains data for the provided time frame.
        """
        # Get list of objects
        object_list = self._nearest(start_time, end_time=end_time)
        # Merge objects
        merged_obj = self.merge_data(object_list)

        return merged_obj

    def merge_data(self, object_list):
        """
        Combine a list of objects into a single FluxData object

        Parameters
        ----------
        object_list : list
            Contains the objects to be merged.

        Returns
        -------
        merged_object : flux_data.FluxData
            The contents of object_list as a single object.
        """
        # If the list is empty, return None
        if len(object_list) == 0:
            return None
        # If the list contains a single item, initialize object
        elif len(object_list) == 1:
            merged_object = object_list[0]
            merged_object.init()
        # If list contains multiple items, merge together.
        else:
            merged_object = object_list[0]
            # Initialize first object
            merged_object.init()
            # Append raw data from other objects
            for this_object in object_list[1:]:
                merged_object.data = merged_object.data.append(
                                            this_object.get_data(),
                                            ignore_index=True
                                        )
            # Set filename to list of all file names
            merged_object.filename = [
                this_object.filename for this_object in object_list]
        
        return merged_object




class EddyProFluxBucket(FluxBucket):
    """
    A bucket of 30-min summary files output from EddyPro.

    """
    def __init__(self, config):
        super().__init__(bucket_object=EddyProFluxData, config=config)
        # dir
        self.dir = os.path.join(config.get('dir'),'summaries')

    # def init(self):
    #     self.objects = self.get_objects()

    # def get_objects(self):
        
    #     # Initialize empty list
    #     objects = []
    #     for file in [f for f in sorted(os.listdir(self.dir)) if f[0] != '.']:
    #         # Get full path of each file in bucket
    #         full_path = os.path.join(self.dir, file)
    #         # Check that path is a file
    #         if os.path.isfile(full_path):
    #             objects.append(self.object_type(full_path, self.config))

    #     return objects



class RawBucket(FluxBucket):
    """
    A bucket of raw 10-Hz data output by EddyPro.

    """
    def __init__(self, config):
        super().__init__(bucket_object=EddyProTower, config=config)
        # dir
        self.dir = config.get('raw_dir')

    # def init(self):
    #     self.objects = self.get_objects()

    # def get_objects(self):
        
    #     # Initialize empty list
    #     objects = []
    #     for file in [f for f in sorted(os.listdir(self.dir)) if f[0] != '.']:
    #         # Get full path of each file in bucket
    #         full_path = os.path.join(self.dir, file)
    #         # Check that path is a file
    #         if os.path.isfile(full_path):
    #             objects.append(self.object_type(full_path, self.config))

    #     return objects

class BiometBucket(FluxBucket):
    """
    A bucket of raw 10-Hz data output by EddyPro.

    """
    def __init__(self, config):
        super().__init__(bucket_object=EddyProTower, config=config)
        # dir
        self.dir = config.get('biomet_dir')

    # def init(self):
    #     self.objects = self.get_objects()

    # def get_objects(self):
        
    #     # Initialize empty list
    #     objects = []
    #     for file in [f for f in sorted(os.listdir(self.dir)) if f[0] != '.']:
    #         # Get full path of each file in bucket
    #         full_path = os.path.join(self.dir, file)
    #         # Check that path is a file
    #         if os.path.isfile(full_path):
    #             objects.append(self.object_type(full_path, self.config))

    #     return objects




    