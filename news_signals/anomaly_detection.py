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
