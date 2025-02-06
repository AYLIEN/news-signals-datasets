# read in list of active Signals tickers which can change slightly era to era
import os
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

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


def retrieve_yfinance_timeseries(tickers, start_date, end_date=None, plot_rsi=False, interval='1d'):
    # Ensure tickers is a list
    if isinstance(tickers, str):
        tickers = tickers.split()
    tickers_str = " ".join(tickers)
    
    # Download the data
    raw_data = yf.download(
        tickers_str, 
        start=str(start_date),
        end=end_date,
        threads=True,
        interval=interval
    )
    
    # If there's only one ticker, the data comes with single-level columns
    # If multiple, yfinance returns a DataFrame with MultiIndex columns.
    if len(tickers) == 1 & plot_rsi == True:
            raw_data['RSI'] = RSI(raw_data['Close'])

    # Plotting
    # Single
    if len(tickers) == 1:
        fig, ax1 = plt.subplots(figsize=(12, 6))
        ax1.plot(raw_data.index, raw_data['Close'], color='blue', label=f"{tickers[0]} Price")
        ax1.grid(True, linestyle='--', alpha=0.6)
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Price (Stock Value)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        if plot_rsi:
            ax2 = ax1.twinx()
            ax2.plot(raw_data.index, raw_data['RSI'], color='red', linestyle='--', label=f"{tickers[0]} RSI")
            ax2.set_ylabel('RSI (Momentum Indicator)', color='red')
            ax2.tick_params(axis='y', labelcolor='red')
        plt.title("Stock Price" + (" and RSI" if plot_rsi else "") + " Over Time")
        fig.tight_layout()
        plt.legend(loc='upper left')
        plt.show()
    else:
        for ticker in tickers:
            price_series = raw_data[('Close', ticker)]
            if plot_rsi:
                rsi_series = RSI(price_series)
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            ax1.plot(raw_data.index, price_series, color='blue', label=f"{ticker} Price")
            ax1.grid(True, linestyle='--', alpha=0.6) 
            ax1.set_xlabel('Date')
            ax1.set_ylabel('Price (Stock Value)', color='blue')
            ax1.tick_params(axis='y', labelcolor='blue')
            if plot_rsi:
                ax2 = ax1.twinx()
                ax2.plot(raw_data.index, rsi_series, color='red', linestyle='--', label=f"{ticker} RSI")
                ax2.set_ylabel('RSI (Momentum Indicator)', color='red')
                ax2.tick_params(axis='y', labelcolor='red')
            plt.title(f"{ticker} Stock Price" + (" and RSI" if plot_rsi else "") + " Over Time")
            fig.tight_layout()
            ax1.legend(loc='upper left')
            if plot_rsi:
                ax2.legend(loc='upper right')
            plt.show()
    return raw_data
