#!/usr/bin/env python

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf

INTERVAL_SYMBOLS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', 
                    '1h', '1d', '5d', '1wk', '1mo', '3mo'] 
def get_daily_data(ticker, start_date=None):
    ticker = ticker.upper()
    if start_date:
        start_date = start_date
        ticker_data = web.DataReader(ticker, 'yahoo', start_date)
        return ticker_data
    else:
        ticker_data = web.DataReader(ticker, 'yahoo')
        return ticker_data

def get_intraday_data(ticker, interval):
    ticker = ticker.upper()
    if interval not in INTERVAL_SYMBOLS:
        raise ValueError('Entered an incorrect interval symbol')
    ticker_data = yf.download(tickers=ticker,
                              period='60d',
                              interval=interval,
                              prepost=False)
    return ticker_data



def main():
    daily_data = get_daily_data('spy')
    intra_data = get_intraday_data('spy', '5m')
    daily_data.to_csv('1d_SPY')
    intra_data.to_csv('5m_SPY')

if __name__=='__main__':
    main()



