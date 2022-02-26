#!/usr/bin/env python

import pandas as pd


def change_n(data, timeperiod=None, **kwargs):
    if isinstance(timeperiod, int):
        period = str(timeperiod)
        col_name = f'change_{period}'
        data[col_name] = data['Close'].pct_change(periods=int(period))
        return data
    if isinstance(timeperiod, list):
        for period in timeperiod:
            if isinstance(period, str):
                period = int(period)
            period = str(timeperiod)
            col_name = f'Change_{period}'
            data[col_name] = data['Close'].pct_change(periods=period)
        return data
                          
def get_vol_direction(data, period=1):
    """Calculates the direction of the volume over the specified period.
    Returns a series containing the values."""
    change = data['Close'].pct_change(periods=1)
    vol_direction = (change / abs(change)) 
    return vol_direction


def vol_tide(data):
    '''Returns a data frame with 3 new columns:
    1. Vol_Direction: just 1 or -1 based on the direction of the change
    2. Dir_to_Vol: multiply Vol_Direction by Volume
    3. Vol_Tide: cumsum for over 1 day of data.
    Args:
        data(pd.DataFrame): expects a dataframe with the columns: Volume, Close
        Volume should be datetime object
    '''
    if 'Vol_Direction' in data.columns:
        data['Dir_to_Vol'] = data['Vol_Direction'] * data['Volume']
        data['Vol_Tide'] = data.groupby([data['Datetime'].dt.date])['Dir_to_Vol'].cumsum()
    else:
        data['Vol_Direction'] = Data.get_vol_direction(data)
        data['Dir_to_Vol'] = data['Vol_Direction'] * data['Volume']
        data['Vol_Tide'] = data.groupby([data['Datetime'].dt.date])['Dir_to_Vol'].cumsum()
    return data
        

# Test by calculating groupby('Date').sum().  This should be equal to 
# the value at the end of the time period.

# period direction function
                          
# period direction placed on volume.

    