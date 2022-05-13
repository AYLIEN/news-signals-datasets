# read in list of active Signals tickers which can change slightly era to era
import os
import yfinance

import pandas as pd

NUMERAI_TICKERS = 'https://numerai-signals-public-data.s3-us-west-2.amazonaws.com/signals_ticker_map_w_bbg.csv'
NAPI_PUBLIC_KEY = os.getenv('NAPI_PUBLIC_KEY')
NAPI_PRIVATE_KEY = os.getenv('NAPI_PRIVATE_KEY')


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


def generate_numerai_signals(ticker, start_date):
    # print(NAPI_PUBLIC_KEY[0])
    # napi = numerapi.SignalsAPI(NAPI_PUBLIC_KEY[0], NAPI_PRIVATE_KEY[0])
    # eligible_tickers = pd.Series(napi.ticker_universe(), name='ticker')
    # print(f"Number of eligible tickers: {len(eligible_tickers)}")
    # ticker_map = pd.read_csv(NUMERAI_TICKERS)
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