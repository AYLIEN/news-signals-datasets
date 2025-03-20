import unittest
from unittest.mock import patch
import pandas as pd

from news_signals.yfinance_utils import retrieve_yfinance_timeseries


class TestRetrieveYFinanceTimeseries(unittest.TestCase):

    @patch('yfinance.download')
    def test_single_ticker_without_rsi(self, mock_yf_download):
        """Test that function returns a DataFrame for a single ticker without RSI"""
        mock_yf_download.return_value = pd.DataFrame({
            'Close': [150, 152, 154, 153, 155]
        }, index=pd.date_range(start="2024-01-01", periods=5))

        df = retrieve_yfinance_timeseries("AAPL", "2024-01-01", "2024-01-05", rsi=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('Close', df.columns)

    @patch('yfinance.download')
    def test_single_ticker_with_rsi(self, mock_yf_download):
        """Test that RSI column is added when plot_rsi=True"""
        mock_yf_download.return_value = pd.DataFrame({
            'Close': [150, 152, 154, 153, 155]
        }, index=pd.date_range(start="2024-01-01", periods=5))

        df = retrieve_yfinance_timeseries("AAPL", "2024-01-01", "2024-01-05", rsi=True)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn('RSI', df.columns)

    @patch('yfinance.download')
    def test_multiple_tickers(self, mock_yf_download):
        """Test multiple tickers with mock yfinance data"""
        mock_yf_download.return_value = pd.DataFrame({
            ('Close', 'AAPL'): [150, 152, 154, 153, 155],
            ('Close', 'MSFT'): [280, 282, 284, 283, 285],
        }, index=pd.date_range(start="2024-01-01", periods=5))

        df = retrieve_yfinance_timeseries(["AAPL", "MSFT"], "2024-01-01", "2024-01-05", rsi=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertIn(('Close', 'AAPL'), df.columns)
        self.assertIn(('Close', 'MSFT'), df.columns)

    @patch('yfinance.download')
    def test_no_data_returned(self, mock_yf_download):
        """Test when yfinance returns an empty DataFrame"""
        mock_yf_download.return_value = pd.DataFrame()

        df = retrieve_yfinance_timeseries("AAPL", "2024-01-01", "2024-01-05", rsi=False)
        self.assertIsInstance(df, pd.DataFrame)
        self.assertTrue(df.empty)


if __name__ == '__main__':
    unittest.main()
