import os
import unittest
from pathlib import Path
import pandas as pd
import numpy as np

from news_signals.anomaly_detection import AnomalyDetector, SigmaAnomalyDetector, BollingerAnomalyDetector


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class TestAnomalyDetector(unittest.TestCase):

    def test_normalize(self):
        s = pd.Series([0, 1, 2, 3, 4])
        norm_history, sigma, _ = AnomalyDetector.normalize(s)

        expected_sigma = s.std()
        self.assertAlmostEqual(sigma, expected_sigma, places=7)

        self.assertTrue((norm_history >= 0).all())
        self.assertTrue((norm_history <= 1).all())

        s_zero = pd.Series([0, 0, 0, 0])
        norm_zero, sigma_zero, max_zero = AnomalyDetector.normalize(s_zero)
        self.assertTrue((norm_zero == 0).all())
        self.assertEqual(sigma_zero, 0.0)
        self.assertEqual(max_zero, 0.0)

    def test_anomaly_weight(self):
        history = pd.Series([4, 5, 6, 5, 4, 6])

        weight = AnomalyDetector.anomaly_weight(history, current=10, sigma_multiple=1, verbose=False)
        self.assertGreater(weight, 0)

        history_zeros = pd.Series([0, 0, 0, 0])

        weight_zeros = AnomalyDetector.anomaly_weight(history_zeros, current=0)
        self.assertEqual(weight_zeros, 0.0)

    def test_history_to_anomaly_ts(self):

        history = pd.Series([10, 12, 13, 20, 22, 23, 25, 2, 3, 5])
        anomaly_ts = AnomalyDetector.history_to_anomaly_ts(history=history)

        self.assertFalse((anomaly_ts == 0).all())
        self.assertEqual(len(anomaly_ts), len(history))

        history_zeros = pd.Series([0, 0, 0, 0])
        anomaly_ts_zeros = AnomalyDetector.history_to_anomaly_ts(history=history_zeros)

        self.assertTrue((anomaly_ts_zeros == 0).all())


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
