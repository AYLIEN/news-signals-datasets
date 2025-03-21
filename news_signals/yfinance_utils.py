# read in list of active Signals tickers which can change slightly era to era
import yfinance as yf
import pandas as pd
from typing import Union, List, Optional

from news_signals.technical_indicators import RSI


def retrieve_yfinance_timeseries(
    tickers: Union[str, List[str]],
    start_date: str,
    end_date: Optional[str] = None,
    rsi: bool = False,
    interval: str = "1d",
) -> pd.DataFrame:

    # Ensuring tickers is a list
    if isinstance(tickers, str):
        tickers = tickers.split()
    tickers_str = " ".join(tickers)

    # Downloading the data
    raw_data = yf.download(
        tickers_str,
        start=str(start_date),
        end=end_date,
        threads=True,
        interval=interval,
    )

    # If there's only one ticker, the data comes with single-level columns
    # If multiple, yfinance returns a DataFrame with MultiIndex columns.
    if len(tickers) == 1 and rsi is True:
        raw_data["RSI"] = RSI(raw_data["Close"])

    # # Plotting
    # # Single ticker
    # if plot and not raw_data.empty:
    #     if len(tickers) == 1:
    #         fig, ax1 = plt.subplots(figsize=(12, 6))
    #         ax1.plot(
    #             raw_data.index,
    #             raw_data["Close"],
    #             color="blue",
    #             label=f"{tickers[0]} Price",
    #         )
    #         ax1.grid(True, linestyle="--", alpha=0.6)
    #         ax1.set_xlabel("Date")
    #         ax1.set_ylabel("Price (Stock Value)", color="blue")
    #         ax1.tick_params(axis="y", labelcolor="blue")
    #         if rsi:
    #             ax2 = ax1.twinx()
    #             ax2.plot(
    #                 raw_data.index,
    #                 raw_data["RSI"],
    #                 color="red",
    #                 linestyle="--",
    #                 label=f"{tickers[0]} RSI",
    #             )
    #             ax2.set_ylabel("RSI (Momentum Indicator)", color="red")
    #             ax2.tick_params(axis="y", labelcolor="red")
    #         plt.title("Stock Price" + (" and RSI" if rsi else "") + " Over Time")
    #         fig.tight_layout()
    #         plt.legend(loc="upper left")
    #         plt.show()
    #     else:
    #         for ticker in tickers:
    #             price_series = raw_data[("Close", ticker)]
    #             if rsi:
    #                 rsi_series = RSI(price_series)

    #             fig, ax1 = plt.subplots(figsize=(12, 6))
    #             ax1.plot(
    #                 raw_data.index, price_series, color="blue", label=f"{ticker} Price"
    #             )
    #             ax1.grid(True, linestyle="--", alpha=0.6)
    #             ax1.set_xlabel("Date")
    #             ax1.set_ylabel("Price (Stock Value)", color="blue")
    #             ax1.tick_params(axis="y", labelcolor="blue")
    #             if rsi:
    #                 ax2 = ax1.twinx()
    #                 ax2.plot(
    #                     raw_data.index,
    #                     rsi_series,
    #                     color="red",
    #                     linestyle="--",
    #                     label=f"{ticker} RSI",
    #                 )
    #                 ax2.set_ylabel("RSI (Momentum Indicator)", color="red")
    #                 ax2.tick_params(axis="y", labelcolor="red")
    #             plt.title(
    #                 f"{ticker} Stock Price" + (" and RSI" if rsi else "") + " Over Time"
    #             )
    #             fig.tight_layout()
    #             ax1.legend(loc="upper left")
    #             if rsi:
    #                 ax2.legend(loc="upper right")
    #             plt.show()
    return raw_data
