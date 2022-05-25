# read in list of active Signals tickers which can change slightly era to era
import os
import yfinance

import pandas as pd


def RSI(prices, interval=14):
    '''Computes Relative Strength Index given a price series and lookback interval
    Modified from https://stackoverflow.com/questions/20526414/relative-strength-index-in-python-pandas
    See more here https://www.investopedia.com/terms/r/rsi.asp'''
    delta = prices.diff()

    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0

    RolUp = dUp.rolling(interval).mean()
    RolDown = dDown.rolling(interval).mean().abs()

    RS = RolUp / RolDown
    RSI = 100.0 - (100.0 / (1.0 + RS))
    return RSI


def retrieve_yfinance_timeseries(ticker, start_date):
    raw_data = yfinance.download(ticker, start=str(start_date), threads=True)
    raw_data =raw_data.rename(columns={'Adj Close': 'price'})
    raw_data['RSI'] = raw_data['price'].transform(lambda x: RSI(x))
    raw_data['RSI_quintile'] = raw_data['RSI'].transform(
        lambda group: pd.qcut(group, 5, labels=False, duplicates='drop'))
    raw_data.dropna(inplace=True)
    num_days = 5
    for day in range(num_days + 1):
        raw_data[f'RSI_quintile_lag_{day}'] = raw_data['RSI_quintile'].transform(lambda group: group.shift(day))
    return raw_data