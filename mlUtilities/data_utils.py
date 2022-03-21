#!/usr/bin/env python

import pandas as pd
import numpy as np
import os, sys


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT_DIR, 'Data')


def get_data(file_name):
    pass


def transform_data(data_df, period='daily', lower_limit=-0.005, upper_limit=0.005, **kwargs):
    '''Expects to receive a pandas dataframe of the base equities data.
    The base equities data should only contain the following columns:
    Open, High, Low, Close, Volume, Adj Close
    Args:
        period(str): should be either "daily" or "intraday"
    '''
    if period == 'intraday':
        try:
            data_df['Datetime'] = pd.to_datetime(data_df['Datetime'])
        except:
            data_df['Datetime'] = pd.to_datetime(data_df['Datetime'], utc=True)
        # Add catch for datetime not coming out as datetime
        if not pd.api.types.is_datetime64_ns_dtype(data_df['Datetime']):
            data_df['Datetime'] = pd.to_datetime(data_df['Datetime'], utc=True)
        data_df = intraday_transform(data_df)
        return data_df
    elif period == 'daily':
        data_df['Date'] = pd.to_datetime(data_df['Date'])
        data_df = daily_transform(data_df, **kwargs)
        data_df.reset_index(drop=True, inplace=True)
        if 'create_targets' in kwargs.keys():
            data_df['Target'] = create_targets(data_df['Change_5'], 
                                               upper_target=upper_limit, 
                                               lower_target=lower_limit)
            data_df['Target'] = data_df['Target'].shift(periods=-1)
        return data_df
    else:
        raise Error ('period argument was not passed to transform_data()')

    
def intraday_transform(data_df):
    '''Used to create the final intraday dataframe dataset.'''
    data_df['Volume'] = min_max_normalize(data_df['Volume'], 0, 5)
    data_df = vol_tide(data_df)
    return data_df


def daily_transform(data_df, **kwargs):
    '''Adds 'Directional_Vol' and 'Vol_Adv_Dec' columns to the dataframe.
        Returns the new dataframe'''
    data_df = directional_volume(data_df)
    data_df['Vol_Adv_Dec'] = data_df['Directional_Vol'].cumsum()
    data_df = change_n(data_df, timeperiod=[1, 3, 5, 10])
    data_df = transform_tide(data_df, **kwargs)
    return data_df
    

def transform_tide(data_df, **kwargs):
    '''Will take a daily dataframe and the EOD Tide Values as inputs.
    Returns a dataframe that includes the columns derived from eod_vol_tide.
    Which includes a re-index to the shorter 60 days of data so that the dataframe and eod tide
    values merge'''
    if 'eod_tide' in kwargs.keys():
        if not isinstance(kwargs['eod_tide'], pd.Series):
            raise Error('eod_tide value must be of type pd.Series')
        eod_vol_tide = kwargs['eod_tide']
        start_date = eod_vol_tide.index[0]
        eod_vol_tide = eod_vol_tide.values
        data_df = data_df.loc[data_df.Date >= str(start_date)]
        data_df['EOD_Vol_Tide'] = (kwargs['eod_tide']).values
        data_df['Tide_Adv_Dec'] = data_df.EOD_Vol_Tide.cumsum()
        data_df['Vol_Diff'] = data_df.Vol_Adv_Dec - data_df.EOD_Vol_Tide
    return data_df
    
    
def get_test_data(intra_data, daily_data, columns):
    intra_data = transform_data(intra_data, period='intraday', columns=columns)
    eod_value = intra_data.groupby([intra_data.Datetime.dt.date])['Dir_to_Vol'].sum()
    daily_data['Volume'] = min_max_normalize(daily_data['Volume'], 0, 5)
    daily_data = transform_data(daily_data, eod_tide=eod_value, create_targets=True)
    short_data = transform_tide(daily_data, eod_tide=eod_vol_tide)
    short_data.dropna(inplace=True)
    target = short_d_data.pop('Target')
    short_data.set_index('Date', inplace=True)
    short_data = short_data[columns]
    return short_data
                            

    
def change_n(data, timeperiod=None, **kwargs):
    '''Calculates change over n period'''
    # Checks if timeperiod is a list or a number
    if isinstance(timeperiod, int):
        period = str(timeperiod)
        col_name = f'change_{period}'
        data[col_name] = data['Close'].pct_change(periods=int(period))
        return data
    if isinstance(timeperiod, list):
        for period in timeperiod:
            period_str = str(period)
            col_name = f'Change_{period_str}'
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
    df['Directional_Vol'] = df['Volume'] * open_close_direction
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
    if not isinstance(data, pd.Series):
        raise ValueError('Data passed to min_max_normalize() should be pd.Series object')
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
    
    
def create_class_targets(df, levels):
    '''Assigsn a class to the target variable.'''
    target_class = pd.DataFrame()
    for col, level in zip(df, levels):
        print(col, level)
        class_values = []
        for val in df[col]:
            if val > level:
                class_values.append('buy')
            elif val < -level:
                class_values.append('sell')
            elif np.isnan(val):
                class_values.append(val)
            else:
                class_values.append('flat')
        target_class[col] = class_values
    return target_class


############### TARGET FUNCTIONS ############### 

def create_targets(array, lower_target=-0.03, upper_target=0.03):
    '''Takes in a pandas Series and returns the shifted class labels for each row of data'''
    if not isinstance(array, pd.Series):
        raise ValueError('pd.Series must be passed into create_targets')
    class_list = []
    for index, value in array.items():
        if value >= upper_target:
            class_list.append('Buy')
        elif value <= lower_target:
            class_list.append('Sell')
        # This else also catches null values
        else:
            class_list.append('Flat')
    return pd.DataFrame(class_list)
        
        
    