#!usr/bin/env python
# -*- coding: utf-8 -*-
#––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––––

__author__ = 'Bryn Morgan'
__contact__ = 'bryn.morgan@geog.ucsb.edu'
__copyright__ = '(c) Bryn Morgan 2021'

__license__ = 'MIT'
__date__ = 'Fri 21 May 21 14:30:35'
__version__ = '1.0'
__status__ = 'initial release'
__url__ = ''

"""

Name:           tower_preproc.py
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

    return df


#%% MAIN
def main():
    
#%%

    # The summaries folder contains the EddyPro output files, described here: 
    # https://www.licor.com/env/support/EddyPro/topics/output-files-full-output.html
    # NOTE: The timestamps correspond to the END of the averaging period.

    # Get list of EddyPro output files with working sonic anemometer (i.e. those 
    # after 21 Jan 2021).
    files = [tab for tab in sorted(os.listdir(os.path.join(filepath,'summaries'))) if tab[0] != '.'][379:]     
    # Import flux data.
    tower = pd.concat([import_tower('summaries/'+file) for file in files],ignore_index=True)



    # Get home directory + add Box filepath
    filepath = os.path.expanduser('~') + '/Box/Dangermond/RamajalTower/'
    # Change directory 
    os.chdir(filepath)




#%%
if __name__ == "__main__":
    main()