#!/usr/bin/env python

import pandas as pd
import os, sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')


def get_data():
    pass

def change_n(data, timeperiod=None, **kwargs):
    '''Calculates change over n period'''
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
    Returns a series containing the values.
    Designed to be used for intraday data as this uses the Close for different periods.
    This is not what one wants when working with daily data."""
    change = data['Close'].pct_change(periods=1)
    vol_direction = (change / abs(change)) 
    return vol_direction


def vol_tide(df):
    '''Returns a data frame with 3 new columns:
    1. Vol_Direction: just 1 or -1 based on the direction of the change
    2. Dir_to_Vol: multiply Vol_Direction by Volume
    3. Vol_Tide: cumsum for over 1 day of data.
    Args:
        data(pd.DataFrame): expects a dataframe with the columns: Volume and Close.
            data should contain intraday data only. 
        Date or Datetime columns should be datetime object.
    '''
    if 'Vol_Direction' in df.columns:
        df['Dir_to_Vol'] = df['Vol_Direction'] * df['Volume']
        df['Vol_Tide'] = df.groupby([df['Datetime'].dt.date])['Dir_to_Vol'].cumsum()
    else:
        df['Vol_Direction'] = get_vol_direction(df)
        df['Dir_to_Vol'] = df['Vol_Direction'] * df['Volume']
        df['Vol_Tide'] = df.groupby([df['Datetime'].dt.date])['Dir_to_Vol'].cumsum()
    return df


'''
def directional_volume(df):
    """Take a daily dataframe and return the dataframe with additional columns."""
    df['Open_Close_Change'] = (df['Close'] - df['Open']) / df['Open']
    df['Open_Close_Direction'] = df['Open_Close_Change'] / abs(df['Open_Close_Change'])
    df['Volume_Direction'] = df['Volume'] * df['Open_Close_Direction']
    return 
'''

        
def directional_volume(df):
    '''Take a daily dataframe and return the dataframe with additional columns.'''
    open_close_change = (df['Close'] - df['Open']) / df['Open']
    open_close_direction = open_close_change / abs(open_close_change)
    df['Directional_Volume'] = df['Volume'] * open_close_direction
    return df


# period direction function
                          
# period direction placed on volume.

    
def min_max_normalize(data, new_min, new_max):
    '''Normalizes data.
    Arguments:
        data(series or array).
        new_min(int)
        new_max(int)
    Returns: series/array containing normalized data
    '''
    data_min = min(data)
    data_max = max(data)
    numerator = data - data_min
    denominator = data_max - data_min
    return numerator / denominator * (new_max - new_min) + new_min
    
    
def mean_normalization(data, column=None):
    if column==None:
        raise Exception('column to normalize not passed into mean_normalization')
    else:
        data[column] = data[column] - data[column].mean()
    return data
    