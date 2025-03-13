import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np


from news_signals.anomaly_detection import SigmaAnomalyDetector, BollingerAnomalyDetector


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class TestSigmaAnomalyDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass

    def test_sigma_anomaly_detector(self):
        anomaly_detector = SigmaAnomalyDetector(sigma_multiple=1.)
        timeseries = pd.Series([10, 20, 30, 40, 0])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 0.0

        timeseries = pd.Series([0, 0, 0, 0, 100])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 100

        timeseries = pd.Series([0])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        with self.assertRaises(AssertionError):
            anomaly_series = anomaly_detector(history, test_series)
        
        timeseries = pd.Series([20, 30, 20, 30, 50])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert np.isclose(anomaly_series.iloc[0], 4.3, atol=0.1)

        timeseries = pd.Series([0, 0, 0, 0, 0, 2, 18])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert np.isclose(anomaly_series.iloc[0], 17.6, atol=0.1)

        timeseries = pd.Series([0, 0, 0, 0, 0, 2])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 2
    
    def test_bollinger_anomaly_detector(self):
        anomaly_detector = BollingerAnomalyDetector(window=3, num_std=2.0)

        # 1) 100 is an anomaly
        timeseries = pd.Series([10, 20, 30, 40, 50, 100])
        history = timeseries[:-1]      # indices [0..4]
        test_series = timeseries[-1:]  # index [5]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 1, "Expected 100 to be anomaly"

        # 2) 50 is within normal range
        timeseries = pd.Series([10, 20, 30, 40, 50])
        history = timeseries[:-1]      # indices [0..3]
        test_series = timeseries[-1:]  # index [4]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 0, "Expected 50 to be normal"

        # 3) 50 is an anomaly when previous are all 10
        timeseries = pd.Series([10, 10, 10, 10, 10, 50])
        history = timeseries[:-1]      # indices [0..4]
        test_series = timeseries[-1:]  # index [5]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 1, "Expected 50 to be anomaly"

        # 4) 60 is within range
        timeseries = pd.Series([10, 20, 30, 40, 50, 60])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 0, "Expected 60 to be normal"

        # 5) 100 is anomaly if previous are near zero
        timeseries = pd.Series([0, 0, 0, 0, 100])
        history = timeseries[:-1]
        test_series = timeseries[-1:]
        anomaly_series = anomaly_detector(history, test_series)
        assert anomaly_series.iloc[0] == 1, "Expected 100 to be anomaly"

if __name__ == '__main__':
    unittest.main()