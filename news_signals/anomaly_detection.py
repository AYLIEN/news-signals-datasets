from .log import create_logger
from abc import abstractmethod


class AnomalyDetector:

    @abstractmethod
    def __call__(self, df_series, **kwargs):
        raise NotImplementedError


class SigmaAnomalyDetector(AnomalyDetector):
    """
    Simplest anomaly detection based upon
    standard deviation.
    """

    def __init__(self, sigma_multiple=1.):
        self.logger = create_logger(__name__)
        self.sigma_multiple = sigma_multiple

    def __call__(self, history, series, **kwargs):
        return self.anomalies_wrt_history(
            history, series, **kwargs
        )

    @staticmethod
    def sigma(series):
        return series.std() if len(series) > 1 else series

    def anomalies_wrt_history(
            self,
            history, test_series,
            sigma_multiple=None
    ):
        if sigma_multiple is None:
            sigma_multiple = self.sigma_multiple
        sigma = self.sigma(history)
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
