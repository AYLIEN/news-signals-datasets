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


class TestSignal(SignalTest):
    def test_date_range(self):
        """
        Test that we can generate a date range
        """
        start = '2020-01-01'
        end = '2020-01-31'
        r = signals.Signal.date_range(start, end)
        assert len(r) == 31

    def test_bdate_range(self):
        """
        Test that we can generate a business date range
        """
        start = '2022-05-01'
        end = '2022-05-08'
        r = signals.Signal.date_range(start, end, freq='D')
        assert len(r) == 8
        r = signals.Signal.date_range(start, end, freq='B')
        assert len(r) == 5

    def test_df(self):
        """
        Test that we can get a single dataframe representation of a signal
        """
        signal = self.aylien_signals()[0]
        df = signal.df
        assert tuple(df.columns) == ('count', 'published_at', 'stories', 'signal_name', 'freq')

    # TODO: this test fails
    # def test_getitem(self):
    #     """
    #     Test that we can get a slice of a signal
    #     """
    #     signal = self.aylien_signals()[0]
    #     start = '1971-01-01'
    #     end = '1971-01-31'
    #     # these dates aren't cached on the signal
    #     df = signal[start:end]
    #     assert len(df) == 0

    def test_signal_start_end_properties(self):
        """
        Test that we can get the start and end dates of a signal
        """
        signal = self.aylien_signals()[0]
        assert len(signal.date_range(signal.start, signal.end)) == len(signal) == len(signal.df)

    def test_freq_property(self):
        """
        Test that we can access the frequency of a signal
        """
        signal = self.aylien_signals()[0]
        assert signal.freq == 'D'


class TestDataframeSignal(SignalTest):

    def test_datetime_index(self):
        """
        Test DataframeSignals index type
        """
        k = list(self.timeseries.keys())[0]
        no_dt_df = aylien_ts_to_df(
            self.timeseries[k],
            dt_index=False,
        )
        with self.assertRaises(AssertionError):
            _ = signals.DataframeSignal(
                name='aylien-signal',
                timeseries_df=no_dt_df
            )
        dt_df = aylien_ts_to_df(
            self.timeseries[k],
            dt_index=True,
            normalize=True,
            freq='D'
        )
        signal = signals.DataframeSignal(
            name='aylien-signal',
            timeseries_df=dt_df
        )
        self.assertEqual(
            'datetime64[ns, UTC]',
            str(signal.timeseries_df.index.dtype),
            'Aylien timeseries should map to UTC '
            + 'timezone-aware dataframe indexes'
        )

    def test_call_dataframe_signal(self):
        """
        Test that we can retrieve a range from df,
        error if the df doesn't have that range
        """
        signal = self.df_signals()[0]
        # dates outside data range of df
        start = '2022-06-01'
        end = '2022-06-08'
        with self.assertRaises(signals.DateRangeNotAvailable):
            _ = signal(start, end)

        # dates inside data range of signal df
        start = '2020-09-29'
        end = '2020-10-05'
        with self.assertRaises(signals.InvalidDateRange):
            _ = signal(end, start)
        ts = signal(start, end).timeseries_df[signal.name]
        assert sum(ts) > 500


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

    def test_call_aylien_signal(self):
        ts_endpoint_mock = MockEndpoint()
        signal = signals.AylienSignal(
            'test-signal',
            params={},
            ts_endpoint=ts_endpoint_mock
        )
        _, ts = next(iter(self.timeseries.items()))
        timeseries_df = aylien_ts_to_df(ts, normalize=True, freq='D')
        # overlapping windows
        w1_start = timeseries_df.index.min()
        w1_end = list(timeseries_df.index)[len(timeseries_df.index) // 2]
        w2_start = list(timeseries_df.index)[len(timeseries_df.index) // 3]
        w2_end = timeseries_df.index.max()

        # dynamically set params to mock endpoint
        payload_1 = {'timeseries_df': timeseries_df[w1_start:w1_end]}
        payload_2 = {'timeseries_df': timeseries_df[w2_start:w2_end]}
        signal.params = payload_1
        _ = signal(w1_start, w1_end)
        signal.params = payload_2
        _ = signal(w2_start, w2_end)
        # the full range is available on the signal now
        self.assertTrue(
            signal.range_in_df(signal.timeseries_df, w1_start, w2_end, freq='D')
        )
        complete_ts = signal(w1_start, w2_end)
        assert ts_endpoint_mock.num_calls == 2, \
            'We called the signal three times but only hit the endpoint twice'
        assert len(complete_ts) == 365

    def test_update(self):
        """
        Test that we can update a signal
        """
        ts_endpoint_mock = MockEndpoint()
        aylien_ts = [
            {"published_at": "2020-01-01T00:00:00Z", "count": 1},
            {"published_at": "2020-01-02T00:00:00Z", "count": 2},
            {"published_at": "2020-01-03T00:00:00Z", "count": 4},
            {"published_at": "2020-01-04T00:00:00Z", "count": 1},
            {"published_at": "2020-01-05T00:00:00Z", "count": 6},
        ]
        timeseries_df = aylien_ts_to_df(aylien_ts, normalize=True, freq='D')
        t1 = list(timeseries_df.index)[0]
        t2 = list(timeseries_df.index)[2]
        t3 = list(timeseries_df.index)[4]
        signal = signals.AylienSignal(
            'test-signal',
            params={'timeseries_df': timeseries_df[t1:t2]},
            ts_endpoint=ts_endpoint_mock
        )
        signal.update(start=t1, end=t2, ts_endpoint=ts_endpoint_mock)
        assert signal.start == t1
        assert signal.end == t2
        signal.params = {'timeseries_df': timeseries_df[t1:t3]}
        signal.update(start=t1, end=t3, ts_endpoint=ts_endpoint_mock)
        assert signal.start == t1
        assert signal.end == t3

    def test_gap_filling(self):
        ts_endpoint_mock = MockEndpoint()
        signal = signals.AylienSignal(
            'test-signal',
            params={},
            ts_endpoint=ts_endpoint_mock
        )
        _, ts = next(iter(self.timeseries.items()))
        timeseries_df = aylien_ts_to_df(ts, normalize=True, freq='D')
        # partially overlapping windows
        w1_start = list(timeseries_df.index)[0]
        w1_end = list(timeseries_df.index)[10]
        w2_start = list(timeseries_df.index)[20]
        w2_end = list(timeseries_df.index)[30]
        w3_start = list(timeseries_df.index)[7]
        w3_end = list(timeseries_df.index)[25]

        # dynamically set params to mock endpoint
        payload_1 = {'timeseries_df': timeseries_df[w1_start:w1_end]}
        payload_2 = {'timeseries_df': timeseries_df[w2_start:w2_end]}
        payload_3 = {'timeseries_df': timeseries_df[w3_start:w3_end]}
        signal.params = payload_1
        _ = signal(w1_start, w1_end)
        signal.params = payload_2
        _ = signal(w2_start, w2_end)
        signal.params = payload_3
        _ = signal(w3_start, w3_end)
        # the full range is available on the signal now
        self.assertTrue(
            signal.range_in_df(signal.timeseries_df, w1_start, w2_end, freq='D')
        )
        self.assertEqual(
            ts_endpoint_mock.num_calls, 3,
            msg='We called the signal three times because the overlap was only partial'
        )

    def test_detect_anomalies(self):
        signal = self.df_signals()[0]
        # dates inside data range of df
        start = '2021-05-01'
        end = '2021-08-05'
        # legacy API (no cache)
        anomaly_signal = signal.anomaly_signal(start, end, cache=False)
        percent_anomaly_days = len(
            anomaly_signal.timeseries_df[
                anomaly_signal.timeseries_df['elon_musk_timeseries-anomalies'] > 1.]
        ) / len(anomaly_signal.timeseries_df)
        date_series = signal.anomaly_dates(start, end)
        assert type(date_series) is pd.Series
        assert date_series.min() > 0., \
            'anomaly weights are positive and start at 0.'

        # Now test treating first part of signal as history, and compute
        # anomalies with respect to that
        # in this pattern we currently assume a history length of 60 days
        signal.timeseries_df = signal.timeseries_df.drop('anomalies', axis=1)
        anomaly_signal = signal.anomaly_signal(cache=True)
        ts_df = anomaly_signal.timeseries_df
        anomaly_scores = ts_df[~ts_df['anomalies'].isna()].anomalies

        history_len = 60
        assert len(signal.timeseries_df.index) - len(anomaly_scores) == history_len

        sigma_multiple = 1.
        percent_anomaly_days = len(
            anomaly_signal.timeseries_df[
                anomaly_signal.timeseries_df['anomalies'] > sigma_multiple]
        ) / len(anomaly_signal.timeseries_df)
        assert 0.25 < percent_anomaly_days < 0.27

    def test_sampling_stories(self):
        stories_endpoint_mock = MockEndpoint()
        signal = signals.AylienSignal(
            'test-signal',
            params={},
            stories_endpoint=stories_endpoint_mock
        )

        # dynamically set mock endpoint payload
        stories_per_tick = 3
        payload = {
            'stories': [
                {'title': 'title', 'body': 'body', 'published_at': '2021-08-02T01:05:00Z'}
                for _ in range(stories_per_tick)
            ]
        }
        signal.params = payload

        # dates inside data range of test df
        start = '2021-08-01'
        end = '2021-08-05'
        signal_with_stories = signal.sample_stories_in_window(start, end, sample_per_tick=False)

        # we called with sample_per_tick=False, so we should have `stories_per_tick` stories total
        assert sum(len(s) for s in signal_with_stories.feeds_df['stories'] if type(s) is list) == stories_per_tick
        # assert type of stories is df with datetime axis
        signal_with_stories = signal.sample_stories_in_window(
            start, end,
            sample_per_tick=True
        )
        date_range = signals.Signal.date_range(start, end)
        assert all(len(s) == stories_per_tick for s in signal_with_stories.feeds_df['stories'] if type(s) is list)
        assert len(signal_with_stories) == len(date_range) - 1

    def test_sampling_stories_for_all_ticks(self):
        signal = self.aylien_signals()[0]
        # dynamically set mock endpoint payload
        stories_per_tick = 3
        payload = {
            'stories': [
                {'title': 'title', 'body': 'body', 'published_at': '2021-08-02T01:05:00Z'}
                for _ in range(stories_per_tick)
            ]
        }
        signal.params = payload

        # we should already have stories for most ticks
        # so the endpoint should not be called for most ticks
        num_ticks_without_stories = len([s for s in signal.df['stories'] if len(s) == 0])
        stories_endpoint_mock = MockEndpoint()
        signal.stories_endpoint = stories_endpoint_mock
        _ = signal.sample_stories()
        assert stories_endpoint_mock.num_calls == num_ticks_without_stories

        # reset mock endpoint
        stories_endpoint_mock.num_calls = 0

        _ = signal.sample_stories(overwrite_existing=True)
        assert stories_endpoint_mock.num_calls == \
            len(signal.date_range(signal.start, signal.end, freq=signal.freq)) - 1

    def test_summarize(self):
        signal, _, _, stories = self.setup_summarization_tests()
        summarizer = summarization.CentralTitleSummarizer()
        raw_summary = summarizer(stories)
        # note cache flag is set to True
        signal_summaries = signal.summarize(
            summarizer, cache_summaries=True
        )
        assert \
            raw_summary.to_dict() == \
            signal_summaries[0] == \
            signal.feeds_df["summary"][0]

    def test_summarizer_params(self):
        signal, _, _, _ = self.setup_summarization_tests()
        summarizer = summarization.TfidfKeywordSummarizer()
        for k in (1, 2, 5):
            summarizer_params = {"top_k": k}
            summaries = signal.summarize(
                summarizer, summarizer_params,
                cache_summaries=True
            )
            summary = summaries[0]
            assert len(summary["summary"].split()) == k
            del signal.feeds_df["summary"]

    def test_add_wikimedia_pageviews_timeseries(self):
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
        signal.add_wikimedia_pageviews_timeseries(
            wikidata_client=MockWikidataClient("wiki-link-placeholder"),
            wikimedia_endpoint=MockRequestsEndpoint(
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
        dtype = signal.timeseries_df.dtypes["wikimedia_pageviews"]
        assert dtype == np.int64

    def test_add_wikipedia_current_events(self):
        ts_endpoint_mock = MockEndpoint()
        aylien_ts = [
            {"published_at": "2023-01-01T00:00:00Z", "count": 1},
            {"published_at": "2023-01-02T00:00:00Z", "count": 2},
            {"published_at": "2023-01-03T00:00:00Z", "count": 4},
            {"published_at": "2023-01-04T00:00:00Z", "count": 1},
            {"published_at": "2023-01-05T00:00:00Z", "count": 6},
        ]
        # WORKING here - fix this test so that it doesn't call the internet
        timeseries_df = aylien_ts_to_df(aylien_ts, normalize=True, freq='D')
        start = '2023-01-01'
        end = '2023-01-05'
        html_path = resources / 'wiki-current-events-portal/example_monthly_page_jan_2023.html'
        example_html = html_path.read_text()
        signal = signals.AylienSignal(
            name='test',
            params={
                'timeseries_df': timeseries_df,
                'entity_ids': ['Q81068910']},
            ts_endpoint=ts_endpoint_mock
        )
        ts_signal = signal(start=start, end=end)
        ts_signal.add_wikipedia_current_events(
            wikidata_client=MockWikidataClient('https://en.wikipedia.org/wiki/COVID-19_pandemic'),
            wikipedia_endpoint=MockRequestsEndpoint(response=example_html)
        )
        assert 'wikipedia_current_events' in ts_signal.feeds_df
        assert not ts_signal.feeds_df['wikipedia_current_events'].isnull().all()
        df = ts_signal.feeds_df
        n = 0
        for events in df['wikipedia_current_events'].values:
            if isinstance(events, list):
                n += 1
                for e in events:
                    assert 'text' in e
                    assert 'date' in e
                    assert 'wiki_links' in e
        assert n > 0

    def test_num_stories_parameter(self):
        # create a mock response
        mock_response = [
            {'title': 'title', 'body': 'body', 'published_at': '2021-08-02T01:05:00Z'}
            for _ in range(5)
        ]
        stories_endpoint_mock = MockRequestsEndpoint(response=mock_response)
        signal = signals.AylienSignal(
            'test-signal',
            params={},
            stories_endpoint=stories_endpoint_mock
        )

        # dates inside data range of test df
        start = '2021-08-01'
        end = '2021-08-05'

        # test with sample_per_tick=True and num_stories=3
        _ = signal.sample_stories_in_window(
            start, end,
            num_stories=3,
            sample_per_tick=True
        )

        assert stories_endpoint_mock.params['per_page'] == 3

        # test with sample_per_tick=False and num_stories=10
        _ = signal.sample_stories_in_window(
            start, end,
            num_stories=10,
            sample_per_tick=False
        )

        assert stories_endpoint_mock.params['per_page'] == 10


class TestWikimediaSignal(SignalTest):

    @classmethod
    def setUpClass(cls):
        cls.pageview_items = {
            'items': [
                {"views": 3, "timestamp": "2023010100"},
                {"views": 4, "timestamp": "2023010200"},
                {"views": 1, "timestamp": "2023010300"},
                {"views": 2, "timestamp": "2023010400"},
                {"views": 3, "timestamp": "2023010500"},
            ]
        }

    def test_create_ts_signal(self):
        signal = signals.WikimediaSignal(
            name='test',
            wikidata_id='Q123',
        )
        start = '2023-01-01'
        end = '2023-01-05'
        ts_signal = signal(
            start=start,
            end=end,
            wikimedia_endpoint=MockRequestsEndpoint(json.dumps(self.pageview_items)),
            wikidata_client=MockWikidataClient('test')
        )
        assert not ts_signal.timeseries_df['wikimedia_pageviews'].isnull().any()

    def test_update(self):
        """
        Test that we can update a signal
        """
        pageviews_items = [
            {"views": 3, "timestamp": "2023010100"},
            {"views": 4, "timestamp": "2023010200"},
            {"views": 1, "timestamp": "2023010300"},
            {"views": 2, "timestamp": "2023010400"},
            {"views": 3, "timestamp": "2023010500"},
        ]
        url_date_format = "%Y%m%d00"
        records = [
            {
                "wikimedia_pageviews": item["views"],
                "timestamp": pytz.utc.localize(datetime.datetime.strptime(item["timestamp"], url_date_format))
            }
            for item in pageviews_items
        ]
        start = records[0]['timestamp']
        end = records[-1]['timestamp']
        timeseries_df = wiki_pageviews_records_to_df(records)
        date_range = pd.date_range(start=start, end=end, freq="D")
        timeseries_df = timeseries_df.reindex(date_range, fill_value=0)

        t1 = list(timeseries_df.index)[0]
        t2 = list(timeseries_df.index)[2]
        t3 = list(timeseries_df.index)[4]
        signal = signals.WikimediaSignal(
            name='test-signal',
            wikidata_id='Q123',
            timeseries_df=timeseries_df[t1:t2]
        )
        signal.update(
            start=t1,
            end=t2,
            wikimedia_endpoint=MockRequestsEndpoint(json.dumps(self.pageview_items)),
            wikidata_client=MockWikidataClient('test'),
        )
        assert signal.start == t1
        assert signal.end == t2
        signal.params = {'timeseries_df': timeseries_df[t1:t3]}
        signal.update(start=t1, end=t3)
        assert signal.start == t1
        assert signal.end == t3

    def test_add_wikipedia_current_events(self):
        html_path = resources / 'wiki-current-events-portal/example_monthly_page_jan_2023.html'
        example_html = html_path.read_text()
        signal = signals.WikimediaSignal(
            name='test',
            wikidata_id='Q81068910',
        )
        start = '2023-01-01'
        end = '2023-01-30'
        ts_signal = signal(start=start, end=end)
        ts_signal.add_wikipedia_current_events(
            wikidata_client=MockWikidataClient('https://en.wikipedia.org/wiki/COVID-19_pandemic'),
            wikipedia_endpoint=MockRequestsEndpoint(response=example_html)
        )
        assert 'wikipedia_current_events' in ts_signal.feeds_df
        assert not ts_signal.feeds_df['wikipedia_current_events'].isnull().all()
        df = ts_signal.feeds_df
        n = 0
        for events in df['wikipedia_current_events'].values:
            if isinstance(events, list):
                n += 1
                for e in events:
                    assert 'text' in e
                    assert 'date' in e
                    assert 'wiki_links' in e
        assert n > 0


class TestWindowDetection(SignalTest):

    def test_window_detection(self):
        """
        test that window detector can take in a timeseries and
        return a list of [(start_time, end_time), ...]
        :return:
        """

        expected_delta = 2
        for signal in self.df_signals():
            ts_len = len(signal.timeseries_df)
            anomaly_window_start = list(signal.timeseries_df.index)[ts_len // 2]
            anomaly_window_end = signal.timeseries_df.index.max()
            signal = signal.anomaly_signal(
                anomaly_window_start,
                anomaly_window_end
            )

            windows, weights = signal.significant_windows()
            for (start_date, end_date), weight in zip(windows, weights):
                self.assertTrue(
                    (arrow.get(end_date) - arrow.get(start_date)).days
                    >= expected_delta)
                # normalize weights is True by default
                assert 0 < weight <= 1.


class TestAggregateSignal(SignalTest):

    def test_call_aggregate_signal(self):
        # assert that the components have the same date ranges
        # and resolutions -- we need to do this at call time
        # some signals are static but others may be updated
        components = [
            signals.DataframeSignal(
                name,
                aylien_ts_to_df(
                    data,
                    dt_index=True,
                    normalize=True,
                    freq='D'
                )
            )
            for name, data in self.timeseries.items()
        ]
        name = '-'.join(self.timeseries.keys())
        agg_signal = signals.AggregateSignal(
            name=name,
            components=components
        )
        # dates inside data range of df
        start = '2020-09-29'
        end = '2020-10-05'
        signal = agg_signal(start, end)
        aylien_values = [
            ts['time_series']
            for ts in self.timeseries.values()
        ]
        daily_counts = [[u['count'] for u in d] for d in zip(*aylien_values)]
        # full daily counts starts one day before
        self.assertTrue(signal.to_series()[0] != sum(daily_counts[0]))
        # assert that values were correctly summed together
        for agg, counts in zip(signal.to_series(), daily_counts[1:]):
            assert agg == sum(counts)


class TestSignalPersistence(SignalTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.db_path = path_to_file / 'test-db.sqlite'
        cls.signal_store = signals.SqliteSignalStore(
            cls.db_path
        )
        cls.temp_signals_dir = Path('test_signals_tmp')
        cls.temp_signals_dir.mkdir(exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        cls.db_path.unlink()
        shutil.rmtree(cls.temp_signals_dir)

    def test_save(self):
        df_signals = self.df_signals()
        for s in df_signals:
            s.save(self.temp_signals_dir)
            assert Path(self.temp_signals_dir / f'{s.id}.static_fields.json').exists()
            assert Path(self.temp_signals_dir / f'{s.id}.timeseries_df.parquet').exists()

    def test_load(self):
        df_signals = self.df_signals()
        for s in df_signals:
            s.save(self.temp_signals_dir)

        loaded_signals = signals_dataset.SignalsDataset.load(
            self.temp_signals_dir
        ).signals
        assert (len(loaded_signals) == len(df_signals))
        aylien_signals = self.aylien_signals()
        # test that the stories dataframe was correctly loaded from disk
        for signal in aylien_signals:
            assert (sum(len(stories) for stories in signal.feeds_df['stories']) > 50)

    def test_sqlitedict_persistence(self):
        """
        Test DataframeSignals index type
        """
        df_signals = self.df_signals()
        ids = []
        for signal in df_signals:
            id = self.signal_store.put(signal)
            ids.append(id)

        loaded_signals = [
            self.signal_store.get(id)
            for id in ids
        ]

        # dates inside data range of dfs
        start = '2020-09-29'
        end = '2020-10-05'
        for s1, s2 in zip(df_signals, loaded_signals):
            s1_df = s1(start, end).timeseries_df
            s2_df = s2(start, end).timeseries_df
            assert s1_df.equals(s2_df)

    def test_get_by_metadata(self):
        """
        test that signals can be retrieved by
        arbitrary metadata key-value pairs
        """
        df_signals = self.df_signals()
        users = []
        entities = []
        ids = []
        for idx, signal in enumerate(df_signals):
            user_id = f'user-{idx}'
            signal.metadata = {
                'user': user_id,
                'entity': signal.name
            }
            _ = self.signal_store.put(signal)
            users.append(user_id)
            entities.append(signal.name)
            ids.append(signal.id)

        # note we know there is only one metadata match for each user and entity from
        # the way we set it up above, thus the [0] to take first item and flatten lists
        signals_by_user = [
            self.signal_store.get_by_metadata({'user': user_id})[0]
            for user_id in users
        ]
        signals_by_entity = [
            self.signal_store.get_by_metadata({'entity': entity})[0]
            for entity in entities
        ]
        signals_by_id = [
            self.signal_store.get(id) for id in ids
        ]
        assert len(df_signals) == len(signals_by_user) \
               == len(signals_by_entity) == len(signals_by_id)

        # dates inside data range of dfs
        start = '2020-09-29'
        end = '2020-10-05'
        for s1, s2, s3, s4 in zip(df_signals, signals_by_id, signals_by_entity, signals_by_user):
            assert all(s1(start, end).timeseries_df.equals(s_(start, end).timeseries_df) for s_ in [s2, s3, s4])
            assert all(s1.id == s_.id for s_ in [s2, s3, s4])

        # test matching on multiple metadata key-value pairs
        match_obj = {
            'user': users[0],
            'entity': entities[0]
        }
        broken_match_obj = {
            'user': users[0],
            'entity': entities[0],
            'aefeea': 'aegagw'
        }
        assert len(self.signal_store.get_by_metadata(match_obj)) == 1
        assert len(self.signal_store.get_by_metadata(broken_match_obj)) == 0


if __name__ == '__main__':
    unittest.main()
