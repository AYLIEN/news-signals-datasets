from .log import create_logger
from abc import abstractmethod


logger = create_logger(__name__, level='INFO')


class AnomalyDetector:

    @abstractmethod
    def __call__(self, df_series, **kwargs):
        raise NotImplementedError

    @staticmethod
    def normalize(history):
        sigma = history.std()
        history = history - sigma
        history = history.clip(lower=0)
        max_ = history.max()
        if max_ == 0:
            # Avoid dividing by zero. If everything is zero, it means there is no anomaly or no volume.
            return history * 0.0, sigma, max_

        history = history / max_
        return history, sigma, max_

    @staticmethod
    def anomaly_weight(history, current, sigma_multiple=1, verbose=False):
        """
        is current value an anomaly based on history?
        :param history:
        :param current:
        :return:
        """
        history, sigma, max_ = AnomalyDetector.normalize(history)
        small_sigma = history.std()
        if max_ == 0:
            weight = 0.0
        else:
            weight = (current - sigma) / max_

        if verbose:
            logger.info("current: %s", current)
            logger.info("max: %s", max_)
            logger.info("sigma: %s", sigma)
            logger.info("small sigma: %s", small_sigma)
            logger.info("weight: %s", weight)
            logger.info("small_sigma * sigma_multiple: %s", small_sigma * sigma_multiple)

        if weight > (small_sigma * sigma_multiple):
            return weight
        else:
            return 0.

    @staticmethod
    def history_to_anomaly_ts(history, sigma_multiple=1):
        history, _, _ = AnomalyDetector.normalize(history)
        small_sigma = history.std()
        history[history < (small_sigma * sigma_multiple)] = 0.
        return history


class SigmaAnomalyDetector(AnomalyDetector):
    """
    Simplest anomaly detection based upon
    standard deviation.
    """

    def __init__(self, sigma_multiple=1., smoothing=1.):
        self.logger = create_logger(__name__)
        self.sigma_multiple = sigma_multiple
        self.smoothing = smoothing

    def __call__(self, history, series, **kwargs):
        return self.anomalies_wrt_history(
            history, series, **kwargs
        )

    @staticmethod
    def sigma(series):
        return series.std() if len(series) > 1 else series

    def anomalies_wrt_history(
            self,
            history, test_series, smoothing=None,
            sigma_multiple=None
    ):
        try:
            assert len(history) > 0, 'history must be non-empty'
        except AssertionError as e:
            raise e
        if sigma_multiple is None:
            sigma_multiple = self.sigma_multiple
        if smoothing is None:
            smoothing = self.smoothing
        history = history + smoothing
        test_series = test_series + smoothing
        if sigma_multiple is None:
            sigma_multiple = self.sigma_multiple
        sigma = self.sigma(history)
        # ensures that sigma shouldn't be zero
        # user can override this by setting smoothing to zero,
        # that could result in infinite anomaly score
        sigma = max(sigma, smoothing)
        mean = history.mean()
        test_series = test_series - mean
        # how many standard deviations away does it need
        # to be outside to be an anomaly?
        multiple = sigma * sigma_multiple

        test_series = test_series / multiple
        # note this positive-only modification makes sense for
        # news, but not for other series types like stocks
        test_series = test_series.clip(lower=0)
        return test_series


class BollingerAnomalyDetector(AnomalyDetector):
    """
    Anomaly detection based on Bollinger Bands.
    Returns Boolean Value (1 for Anomaly, 0 otherwise)
    """

    def __init__(self, window=20, num_std=2.0):
        self.logger = create_logger(__name__)
        self.window = window
        self.num_std = num_std

    def __call__(self, history, series, **kwargs):
        return self.anomalies_wrt_history(history, series, **kwargs)

    def anomalies_wrt_history(
            self,
            history, test_series,
            window=None, num_std=None
    ):
        if window is None:
            window = self.window
        if num_std is None:
            num_std = self.num_std

        if len(history) < window:
            raise ValueError(f'history must be at least {window} points long')

        rolling_mean = history.rolling(window=window).mean()
        rolling_std = history.rolling(window=window).std()
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        # Incase of reindexing, forward fill the values
        upper_band = upper_band.reindex(test_series.index, method='ffill')
        lower_band = lower_band.reindex(test_series.index, method='ffill')
        anomalies = (test_series > upper_band) | (test_series < lower_band)
        return anomalies.astype(int)
