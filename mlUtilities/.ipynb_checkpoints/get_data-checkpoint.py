#!/usr/bin/env python

import pandas as pd
import pandas_datareader.data as web
import yfinance as yf
import argparse
import traceback

INTERVAL_SYMBOLS = ['1m', '2m', '5m', '15m', '30m', '60m', '90m', 
                    '1h', '1d', '5d', '1wk', '1mo', '3mo'] 


def Parser():
    """Create Parser.
    returns: parser.parsearguments() object"""
    program_description = 'run as: python get_data.py <ticker symbol>'
    parser = argparse.ArgumentParser(description='Download Stock OHLCV')
    parser.add_argument("TICKER",
                        metavar = "<TickerSymbol>",
                        type=str,
                        help = 'Enter ticker you wish to get data for')
    args = parser.parse_args()
    return args


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
    parser = Parser()
    try:
        if parser:
            ticker = parser.TICKER.upper()
            print('Getting data for ', ticker)
            daily_data = get_daily_data(ticker)
            intra_data = get_intraday_data(ticker, '5m')
            daily_data.to_csv(f'{ticker}_1d_SPY.csv')
            intra_data.to_csv(f'{ticker}_5m_SPY.csv')
        return None
    except Exception:
        print(traceback.format_exc())

if __name__=='__main__':
    main()



