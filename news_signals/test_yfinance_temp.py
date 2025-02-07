import os
import unittest
from pathlib import Path
import json
import shutil
import pytz

import arrow
import datetime
import pandas as pd
import numpy as np

from news_signals.data import aylien_ts_to_df
from news_signals import signals, signals_dataset
from news_signals import summarization
from news_signals.exogenous_signals import wiki_pageviews_records_to_df
from news_signals.log import create_logger


logger = create_logger(__name__)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class SignalTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.timeseries = {}
        for ts in sorted(resources.glob('*timeseries.json')):
            name = ts.parts[-1].split('.')[0]
            with open(ts) as f:
                data = json.load(f)
            cls.timeseries[name] = data
        cls.resources = resources

    @classmethod
    def df_signals(cls):
        df_sigs = []
        # note assumption that cls.setUpClass already ran
        for name, data in cls.timeseries.items():
            df_sigs.append(
                signals.DataframeSignal(
                    name=name,
                    timeseries_df=aylien_ts_to_df(
                        data,
                        dt_index=True,
                        normalize=True,
                        freq='D'
                    )
                )
            )
        return df_sigs

    @classmethod
    def aylien_signals(cls):
        return signals.Signal.load(resources / 'nasdaq100_sample_dataset')

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass


class MockEndpoint:
    def __init__(self):
        self.num_calls = 0

    def __call__(self, payload):
        self.num_calls += 1
        # mock timeseries endpoint
        if type(payload.get('timeseries_df', None)) is pd.DataFrame:
            # map dataframe to look like Aylien API response
            ts = [
                {'count': 10, 'published_at': dt}
                for dt in payload['timeseries_df'].index
            ]
            logger.info(f'returning {len(ts)} rows from mock endpoint')
            return {'time_series': ts}
        elif type(payload.get('stories', None)) is not None:
            # map stories to look like Aylien API response
            logger.info(f'returning {len(payload["stories"])} stories from mock endpoint')
            return payload['stories']

        return None


class MockWikidataClient:
    def __init__(self, wikipedia_link):
        self.wikipedia_link = wikipedia_link

    def __call__(self, wikidata_id):
        return {
            "sitelinks": {
                "enwiki": {
                    "url": self.wikipedia_link
                }
            },
        }


class MockRequestsEndpoint:
    def __init__(self, response):
        self.response = response
        self.params = {}
        self.headers = {}
        self.url = ""

    def __call__(
        self,
        params: dict={},
        headers: dict={},
        url: str="",
    ):
        self.params = params
        self.headers = headers
        self.url = url
        return self.response


class TestAylienSignal(SignalTest):

    @classmethod
    def setup_summarization_tests(cls):
        with open(resources / "tesla_stories.json") as f:
            stories = json.load(f)
        stories_endpoint_mock = MockEndpoint()
        signal = signals.AylienSignal(
            'test-signal',
            params={},
            stories_endpoint=stories_endpoint_mock
        )
        signal.params = {'stories': stories}
        start = '2022-12-01'
        end = '2022-12-02'
        signal.sample_stories_in_window(
            start, end, sample_per_tick=True
        )
        return signal, start, end, stories


    def test_add_yfinance_market_timeseries(self):
        aylien_ts = [
            {"published_at": "2020-01-01T00:00:00Z", "count": 1},
            {"published_at": "2020-01-02T00:00:00Z", "count": 2},
            {"published_at": "2020-01-03T00:00:00Z", "count": 4},
            {"published_at": "2020-01-04T00:00:00Z", "count": 1},
            {"published_at": "2020-01-05T00:00:00Z", "count": 6},
        ]
        timeseries_df = aylien_ts_to_df(aylien_ts, normalize=True, freq='D')
        signal = signals.AylienSignal(
            'test-signal',
            params={"entity_ids": ["Q42"]},
            timeseries_df=timeseries_df
        )

        signal.add_yfinance_market_timeseries()
        import ipdb; ipdb.set_trace()

        signal.add_wikimedia_pageviews_timeseries(
            wikidata_client=MockWikidataClient("wiki-link-placeholder"),
            wikimedia_endpoint = MockRequestsEndpoint(
                response=json.dumps(
                    {
                        "items": [
                            {"views": 3, "timestamp": "2020010100"},
                            {"views": 4, "timestamp": "2020010200"},
                            {"views": 1, "timestamp": "2020010300"},
                            {"views": 2, "timestamp": "2020010400"},
                            {"views": 3, "timestamp": "2020010500"},
                        ]
                    }
                )
            )
        )

        assert "wikimedia_pageviews" in signal.timeseries_df.columns
        dtype =  signal.timeseries_df.dtypes["wikimedia_pageviews"]
        assert dtype == np.int64


if __name__ == '__main__':
    unittest.main()
