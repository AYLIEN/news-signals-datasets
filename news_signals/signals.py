import copy
import os
import sys
from abc import abstractmethod
from collections import Counter, defaultdict
from typing import List, Optional
import json
import base64
from pathlib import Path
import tqdm
import pandas as pd
import arrow
import datetime
from sqlitedict import SqliteDict
from dataclasses import dataclass

from .newsapi import retrieve_timeseries, retrieve_stories
from .data import arrow_to_aylien_date
from .log import create_logger
from .data import aylien_ts_to_df, datetime_to_aylien_str
from .anomaly_detection import SigmaAnomalyDetector
from .aql_builder import params_to_aql
from .yfinance_utils import retrieve_yfinance_timeseries
from .summarization import Summarizer
from .semantic_filters import SemanticFilter
from .exogenous_signals import (
    wikidata_id_to_wikimedia_pageviews_timeseries,
    wikidata_id_to_current_events
)

logger = create_logger(__name__)


class DateRangeNotAvailable(Exception):
    pass


class InvalidDateRange(Exception):
    pass


class UnknownFrequencyArgument(Exception):
    pass


class NoStoriesException(Exception):
    pass


class WikidataIDNotFound(Exception):
    pass


class Signal:
    """
    Signals have names
    Signal timeseries are stored in signal.timeseries_df
    Signal feeds are stored in signal.feeds_df
    Both feeds and timeseries are pandas DataFrames, with a DatetimeIndex
    Feeds and timeseries associated with signals are named, that's how
    users know the semantics of the timeseries and feeds.

    Signals have periods ("ticks") - but these may be implicit, e.g. if
    using the `infer_freq` method of pandas DatetimeIndex.

    """
    def __init__(self, name, metadata=None, timeseries_df=None, feeds_df=None, ts_column='count'):
        self.name = name
        if metadata is None:
            metadata = dict()
        self.metadata = metadata

        if timeseries_df is not None:
            if type(timeseries_df) is pd.Series:
                timeseries_df = timeseries_df.to_frame(name=name)
                ts_column = name
            self.assert_df_index_type(timeseries_df)

        if feeds_df is not None:
            self.assert_df_index_type(feeds_df)

        self.timeseries_df = timeseries_df
        self.feeds_df = feeds_df
        self.ts_column = ts_column

    @staticmethod
    def assert_df_index_type(df):
        assert hasattr(df.index, 'tz'), \
            'we expect dataframes with timezone-aware index dtypes'
        assert str(df.index.tz) == 'UTC', \
            'we expect dataframes with timezone-aware index dtypes in UTC timezone'

    @abstractmethod
    def __call__(self, start, end, freq='D'):
        """
        Return a signal's data between start time and
        end time

        Signals may be transformations like anomaly detection, or
        they may be compositions like aggregation or decompositions like
        structural timeseries models.
        In general we view signal processing as a directed graph from inputs
        to outputs. Users subscribe to nodes in the graph and define notification
        conditions.

        See pandas timeseries documentation for more information on the
        interface design
        https://pandas.pydata.org/pandas-docs/dev/user_guide/timeseries.html

        Note in particular the difference between `date_range` (calendar day)
        and `bdate_range` (business day) which may be relevant in the context
        of market signal analysis.
        See pandas Offset aliases for info on how date ranges are generated
        https://pandas.pydata.org/pandas-docs/dev/user_guide/timeseries.html#timeseries-offset-aliases

        :param start_date: datetime
        :param end_date: datetime
        :return: new Signal instance with the result of calling this signal
        """
        raise NotImplementedError

    @property
    def df(self):
        """
        Join the timeseries and feeds dataframes
        """
        if self.timeseries_df is None:
            df = self.feeds_df
        elif self.feeds_df is None:
            df = self.timeseries_df
        else:
            df = self.timeseries_df.join(
                self.feeds_df,
                how='left'
            )

        # static metadata about the signal
        df['signal_name'] = self.name
        df['freq'] = self.infer_freq()
        return df

    @property
    def start(self):
        return self.df.index.min()

    @property
    def end(self):
        return self.df.index.max()

    @abstractmethod
    def inputs(self):
        """
        return the inputs to this signal
        this let us easily traverse the signal graph
        to give insight into the path that a signal takes
        """
        raise NotImplementedError

    @staticmethod
    def normalize_timestamp(ts, freq):
        """
        rounds timestep down to the nearest interval tick
        """
        ts = pd.Timestamp(ts).floor(freq=freq)
        if ts.tzname() is None:
            ts = ts.tz_localize(tz='UTC')
        return ts

    @staticmethod
    def date_range(start, end, freq='D', tz='UTC', **kwargs):
        """
        Note that pandas also supports more flexible date ranges
        for holidays, etc... if we eventually need that.

        :returns: DatetimeIndex
        """
        r = pd.date_range(
            start, end,
            freq=freq, tz=tz, inclusive='both', **kwargs
        )
        if len(r) == 0:
            raise InvalidDateRange(
                'Signals do not support 0-length or negative date ranges'
            )
        return r

    @staticmethod
    def range_in_df(df, start, end, freq='D'):
        if df is None:
            return False
        r = Signal.date_range(start, end, freq=freq)
        ts = df[start:end]
        if len(ts) != len(r):
            return False
        else:
            return True

    def to_series(self):
        if getattr(self, 'timeseries_df', None) is not None:
            return self.timeseries_df[self.ts_column]
        else:
            raise NotImplementedError(
                'to_series() is not implemented for this signal type'
            )

    def infer_freq(self):
        try:
            if getattr(self, 'timeseries_df', None) is not None:
                return pd.infer_freq(self.timeseries_df.index)
            elif getattr(self, 'feeds_df', None) is not None:
                return pd.infer_freq(self.feeds_df.index)
            else:
                raise NotImplementedError(
                    'infer_freq() is not implemented for this signal type'
                )
        except ValueError:
            logger.warning(
                f'Could not infer frequency for signal {self.name}, '
                'this may be because the signal has no data, returning default freq = \'D\''
            )
            return 'D'

    def significant_windows(
            self, min_value=1., min_delta=3, window_range=1,
            format='datetime', normalize_weights=True
    ):
        freq = self.infer_freq()
        freq_attr = 'days'
        if freq == 'H':
            freq_attr = 'hours'
        series = self.to_series()
        if series.mean() > min_value:
            logger.warning(
                'The mean of the series is larger than the threshold value '
                'for significance, this is probably because the signal has not '
                'been transformed to an anomaly signal, and is likely to be an error'
            )
        windows = []
        weights = []
        prev = None
        for date, value in zip(series.index, series):
            current = arrow.get(date)
            if value >= min_value:
                if prev is not None and getattr(current - prev, freq_attr) < min_delta:
                    windows[-1][1] = current
                    weights[-1] = max(weights[-1], value)
                else:
                    windows.append([current, current])
                    weights.append(value)
                prev = current

        if len(weights) and normalize_weights:
            max_w = max(weights)
            weights = [w / max_w for w in weights]

        # widen windows by shifting start and end
        windows = \
            [(sd.shift(days=-window_range), ed.shift(days=+window_range))
             for sd, ed in windows]
        if format == 'datetime':
            return [(sd.datetime, ed.datetime) for sd, ed in windows], weights
        else:
            # isoformat
            return [
                (datetime_to_aylien_str(sd.datetime),
                 datetime_to_aylien_str(ed.datetime))
                for sd, ed in windows
            ], weights

    def __len__(self):
        if getattr(self, 'timeseries_df', None) is not None:
            return len(self.timeseries_df)
        elif getattr(self, 'feeds_df', None) is not None:
            return len(self.feeds_df)
        else:
            raise NotImplementedError(
                'len() is not implemented for this signal type'
            )

    def anomaly_dates(
            self, start, end, freq='D',
    ):
        """
        return the dates in the interval that were anomalous, with
        their weights
        :return: pd.Series with datetime index, values are anomaly weights
        """
        signal = self.anomaly_signal(start, end, freq=freq)
        return signal.timeseries_df[signal.timeseries_df[signal.ts_column] > 0.][signal.ts_column]

    def anomaly_signal(
            self, start=None, end=None, freq='D',
            history_length=1,
            history_interval='months',
            cache=True,
            overwrite_existing=False,
            detector=None):
        """
        Anomaly detection methods expect a minimum amount of history,
        or function is not idempotent wrt dates (same date will have different scores
        over time)

        Anomaly detection methods have a threshold
        For aggregate signals, we may sometimes want to set _different_ thresholds

        """

        # if user didn't supply start and end, we want signal to have enough data that
        # we can take the first part and use it to compute necessary stats to do the
        # anomaly transformation on the rest of the signal
        if not overwrite_existing and self.timeseries_df is not None and 'anomalies' in self.timeseries_df.columns:
            return self

        if start is None:
            ts_begin = self.timeseries_df.index.min()
            ts_end = self.timeseries_df.index.max()
            dates = self.date_range(ts_begin, ts_end)
            if freq == 'D' and len(dates) > 2:
                history_length = min(len(dates) // 2, 60)
                history_interval = 'days'
                # the index where the anomaly transformation will start from
                start_idx = history_length
                start = self.timeseries_df.index[start_idx]
                end = ts_end
            else:
                raise NotImplementedError(
                    'History length imputation is only supported for daily ticks, '
                    'and signals need at least two ticks'
                )

        if detector is None:
            detector = SigmaAnomalyDetector()
        shift_kwargs = {history_interval: -history_length}
        # since history_start is now utc datetime we need to convert
        # the others to utc datetime as well or pandas date range
        # won't work, that's why we cast with arrow.get
        history_start = arrow.get(start).shift(**shift_kwargs).datetime
        start = arrow.get(start).datetime
        end = arrow.get(end).datetime
        # get history, then compute anomalies wrt history
        full_signal = self.__call__(history_start, end, freq)
        history = full_signal(history_start, start, freq)
        series = full_signal(start, end, freq).to_series()
        series = detector(history.to_series(), series)

        # side effect
        if cache:
            series.name = 'anomalies'
            anomaly_df = series.to_frame()
            if overwrite_existing and 'anomalies' in self.timeseries_df:
                del self.timeseries_df['anomalies']
            self.timeseries_df = self.timeseries_df.join(anomaly_df, how='left')
            return self
        else:
            # legacy, deprecate
            return DataframeSignal(f'{self.name}-anomalies', timeseries_df=series)

    def __repr__(self):
        signal_dict = self.to_dict()
        signal_dict['type'] = str(signal_dict['type'])
        if getattr(self, 'timeseries_df', None) is not None:
            signal_dict['timeseries_df_columns'] = \
                str(self.timeseries_df.columns.tolist())
            del signal_dict['timeseries_df']
        else:
            signal_dict['timeseries_df_columns'] = None
        if getattr(self, 'feeds_df', None) is not None:
            signal_dict['feeds_df_columns'] = \
                str(self.feeds_df.columns.tolist())
            del signal_dict['feeds_df']
        else:
            signal_dict['feeds_df_columns'] = None
        return json.dumps(signal_dict, indent=2)

    def plot(self, *args, **kwargs):
        if getattr(self, 'timeseries_df', None) is not None:
            return self.timeseries_df.plot(*args, **kwargs)
        else:
            raise NotImplementedError(
                'plot() is not implemented for this signal type'
            )

    def __str__(self):
        return self.__repr__()

    def __getattr__(self, name):
        """
        Try to delegate to the underlying timeseries_df if the attribute
        is not found on the signal itself.
        """
        try:
            return getattr(self.timeseries_df, name)
        except AttributeError:
            raise AttributeError(
                f"type object '{type(self)}' has no attribute '{name}'"
            )

    def __getitem__(self, subscript):
        """
        Delegate slicing semantics to pandas
        """
        return self.df.__getitem__(subscript)

    @staticmethod
    def from_dict(data):
        # expect a `type` arg that tells us the kind of signal we're loading
        signal_type = data['type']
        if type(signal_type) is str:
            signal_type = getattr(sys.modules[__name__], signal_type)
        args = dict(**data)
        # remap legacy `df` to `timeseries_df`

        if 'df' in args:
            args['timeseries_df'] = args.pop('df')
        # remap legacy `stories_df` to `feeds_df`
        if 'stories_df' in args:
            args['feeds_df'] = args.pop('stories_df')
        return signal_type.from_dict(args)

    def save(self, datadir):
        """
        Save a signal to disk
        """
        datadir = Path(datadir)
        signal_id = self.id
        signal_dict = self.to_dict()
        static_fields = {
            k: v
            for k, v in signal_dict.items()
            if type(v) is not pd.DataFrame
        }
        # make type json serializable
        static_fields['type'] = type(self).__name__
        signal_config_file = datadir / f'{signal_id}.static_fields.json'
        with open(signal_config_file, 'w') as out:
            out.write(json.dumps(static_fields, indent=2))

        # "time indexed columns" are ones that are in dfs in the original signal
        for k, v in signal_dict.items():
            if type(v) is pd.DataFrame:
                v.to_parquet(datadir / f'{signal_id}.{k}.parquet', index=True)
        return signal_config_file

    @staticmethod
    def load_from_signal_config(signal_config_path):
        signal_config_path = Path(signal_config_path)
        assert signal_config_path.is_file(), f'signal config {signal_config_path} not found'
        with open(signal_config_path) as f:
            signal_config = json.load(f)
        signal_id = str(signal_config_path.name).split('.')[0]
        # load signal dataframes from parquet files
        parent_dir = signal_config_path.parent
        df_paths = [p.name for p in parent_dir.glob(f'{signal_id}.*.parquet')]
        for df_path in df_paths:
            df_key = str(df_path).split('.')[1]
            df = pd.read_parquet(parent_dir / df_path)
            # if the df index is not already datetime64, cast it
            if not df.index.inferred_type == "datetime64":
                df.index = pd.to_datetime(df.index)
            signal_config[df_key] = df
        return Signal.from_dict(signal_config)

    @staticmethod
    def load(signals_path):
        signals_path = Path(signals_path)
        if os.path.isdir(signals_path):
            signals_dir = Path(signals_path)

            static_config_paths = signals_dir.glob('*.static_fields.json')
            signals = []
            for signal_config_path in static_config_paths:
                signals.append(Signal.load_from_signal_config(signal_config_path))
            return signals
        else:
            assert str(signals_path).endswith('.static_fields.json'), f'expected a static_fields.json file, got {signals_path}'
            assert signals_path.is_file(), f'signal config {signals_path} not found'
            return Signal.load_from_signal_config(signals_path)

    @property
    def id(self):
        """
        Generate a unique id for this signal
        by leveraging the `name` and `metadata` fields, the user
        can control how the id is generated, and thus control the equality
        semantics of signals.
        """
        id_str = json.dumps(
            {
                'name': self.name,
                'metadata': self.metadata
            }
        )
        return base64.b64encode(id_str.encode()).decode()

    @property
    def start(self):
        """
        Return the start timestamp of the signal
        """
        return self.timeseries_df.index.min()

    @property
    def end(self):
        """
        Return the start timestamp of the signal
        """
        return self.timeseries_df.index.max()

    @property
    def freq(self):
        """
        Return the frequency of the signal
        """
        return pd.infer_freq(self.timeseries_df.index)


class DataframeSignal(Signal):
    """
    Holds static data in a dataframe with a datetime index
    """
    def __init__(
        self, name,
        timeseries_df, metadata=None,
        feeds_df=None, ts_column='count'
    ):
        super().__init__(
            name,
            metadata=metadata, timeseries_df=timeseries_df, feeds_df=feeds_df, ts_column=ts_column
        )

    def to_dict(self):
        return {
            'type': type(self),
            'name': self.name,
            'metadata': self.metadata,
            'timeseries_df': self.timeseries_df,
            'feeds_df': self.feeds_df,
            'ts_column': self.ts_column
        }

    @staticmethod
    def from_dict(data):
        return DataframeSignal(
            name=data['name'],
            metadata=data['metadata'],
            timeseries_df=data['timeseries_df'],
            feeds_df=data['feeds_df'],
            ts_column=data['ts_column'],
        )

    def __call__(self, start, end, freq='D'):
        start = self.normalize_timestamp(start, freq)
        end = self.normalize_timestamp(end, freq)
        ts = self.timeseries_df.loc[start:end][self.ts_column]
        expected_range = self.date_range(start, end, freq=freq)
        if len(ts) != len(expected_range):
            if len(ts) > 2 and abs(len(ts) - len(expected_range)) <= 3:
                logger.warning(
                    'The length timeseries is  greater/less than expected, '
                    'this may be due to pandas date_range `inclusive` kwarg, make sure '
                    'you are cool with the length of the timeseries')
            else:
                raise DateRangeNotAvailable(
                    'the expected date range does not match the dataframe index\n'
                    f'len(expected): {len(expected_range)}, len signal: {len(ts)}\n'
                    f'min expected: {expected_range.min()}, max expected: {expected_range.max()}\n'
                    f'min in signal: {ts.index.min()} max in signal: {ts.index.max()}'
                )
        return DataframeSignal(name=self.name, timeseries_df=ts)


class AylienSignal(Signal):

    """
    An Aylien signal wraps News API query to the
    Timeseries endpoint and stores its output
    """
    def __init__(
        self, name,
        metadata=None,
        timeseries_df=None,
        feeds_df=None,
        params=None,
        aql=None,
        ts_column='count',
        ts_endpoint=retrieve_timeseries,
        stories_endpoint=retrieve_stories
    ):
        super().__init__(
            name,
            metadata=metadata,
            timeseries_df=timeseries_df,
            feeds_df=feeds_df,
            ts_column=ts_column
        )
        if params is None and aql is None:
            raise NotImplementedError('one of params or aql must be given')

        if params is not None:
            if 'language' not in params:
                params['language'] = 'en'
            if 'sort_by' not in params:
                params['sort_by'] = 'relevance'
            # warn user if start_date end_date in params,
            # because these will be overwritten at query time
            if 'published_at.start' in params or 'published_at.end' in params:
                logger.warning(
                    'published_at.start and/or published_at.end were provided in params, '
                    + 'but these fields will be overwritten '
                    + 'when the signal is called.'
                )
        else:
            params = {}

        if aql is None:
            aql = params_to_aql(params)
        else:
            if len(params):
                logger.warning(
                    'both aql and params were given - any params used for '
                    'generating aql will have no effect '
                    'and the aql will be used directly'
                )
        self.params = params
        self.aql = aql
        self.ts_endpoint = ts_endpoint
        self.stories_endpoint = stories_endpoint

    def to_dict(self):
        return {
            'type': type(self),
            'name': self.name,
            'metadata': self.metadata,
            'params': self.params,
            'aql': self.aql,
            'timeseries_df': self.timeseries_df,
            'feeds_df': self.feeds_df,
            'ts_column': self.ts_column
        }

    @staticmethod
    def from_dict(data):
        return AylienSignal(
            name=data['name'],
            metadata=data['metadata'],
            params=data['params'],
            aql=data['aql'],
            timeseries_df=data['timeseries_df'],
            feeds_df=data['feeds_df'],
            ts_column=data['ts_column'],
        )

    def __call__(self, start, end, freq='D'):
        start = self.normalize_timestamp(start, freq)
        end = self.normalize_timestamp(end, freq)
        if freq not in ['D', 'H']:
            # currently we only support daily and hourly ticks
            # on Aylien timeseries
            raise UnknownFrequencyArgument

        self.update(start=start, end=end, freq=freq)

        return self

    @staticmethod
    def pd_freq_to_aylien_period(freq):
        if freq == 'D':
            return '+1DAY'
        elif freq == 'H':
            return '+1HOUR'
        else:
            raise UnknownFrequencyArgument

    def update(self, start=None, end=None, freq='D', ts_endpoint=None):
        """
        This method should eventually update all of the data in the signal, not just
        the timeseries_df. This is a work in progress.

        Side effect: we may have other already data in the state, we want to upsert
        any new data while retaining the existing information as well
        :param start: datetime
        :param end: datetime
        """
        if end is None:
            end = self.normalize_timestamp(datetime.datetime.now(), freq)
        # if start is None, we look up to 30 days ago
        if start is None:
            default_interval = self.normalize_timestamp(
                end - datetime.timedelta(days=30),
                freq
            )
            current_end = self.timeseries_df.index.max()
            if current_end > default_interval:
                start = current_end
            else:
                start = default_interval
                logger.warning(
                    f'When updating signal, signal was either empty or the maximum, '
                    f'end date was more than 30 days ago, so we are using '
                    f'default update interval of 30 days --> {start} to {end}'
                )
        if ts_endpoint is None:
            ts_endpoint = self.ts_endpoint

        # first check if we already have this time range,
        # if so, we don't need to query again
        range_exists = \
            self.range_in_df(
                self.timeseries_df, start, end,
                freq=freq
            )
        if not range_exists:
            # update start and end to just get the data we don't have
            # we're only going to be clever about extending to the right,
            # if user wants historical data (before the data we already have),
            # everything's getting retrieved
            if self.timeseries_df is not None and start in self.timeseries_df.index:
                r = Signal.date_range(start, end, freq=freq)
                # find the first index that doesn't match
                for dt, idx_dt in zip(r, self.timeseries_df[start:].index):
                    if dt != idx_dt:
                        start = dt
                        break
            period = self.pd_freq_to_aylien_period(freq)
            aylien_ts_df = self.query_news_signals(start, end, period, ts_endpoint)
            if self.timeseries_df is None:
                self.timeseries_df = aylien_ts_df
            else:
                # note new values _do not_ overwrite old ones if index values
                # are the same
                self.timeseries_df = self.timeseries_df.combine_first(aylien_ts_df)

    def make_query(self, start, end, period='+1DAY', **kwargs):
        _start = arrow_to_aylien_date(arrow.get(start))
        _end = arrow_to_aylien_date(arrow.get(end))
        params = copy.deepcopy(self.params)
        params['published_at.start'] = _start
        params['published_at.end'] = _end
        params['period'] = period
        if self.aql is not None:
            params['aql'] = self.aql
        params.update(kwargs)
        return params

    def query_news_signals(self, start, end, period, ts_endpoint=None):
        if ts_endpoint is None:
            ts_endpoint = self.ts_endpoint
        params = self.make_query(start, end, period=period)
        aylien_ts = ts_endpoint(params)
        ts_df = aylien_ts_to_df(
            aylien_ts, dt_index=True
        )
        return ts_df

    def create_aylien_dataset(self, start, end):
        _ = self.__call__(start, end)
        _ = self.sample_stories_in_window(
            start, end, sample_per_tick=True
        )
        return self.to_dict()

    def sample_stories(self, num_stories=10, **kwargs):
        """
        sample stories for every tick of this signal
        """
        self.sample_stories_in_window(
            self.start, self.end,
            sample_per_tick=True, freq=self.freq, num_stories=num_stories,
            **kwargs
        )
        return self

    def filter_stories(self, filter_model: SemanticFilter, delete_filtered: bool = True, **kwargs) -> Signal:
        """
        Filter stories in the signal using a semantic model, adding a column `matching_scores` to the feeds_df
        """
        for index, tick_stories in self.feeds_df['stories'].items():
            filtered_stories = []
            for story in tick_stories:
                keep = filter_model(story)
                if story.get('filter_model_scores') is None:
                    story['filter_model_outputs'] = [(filter_model.name, keep)]
                if keep or not delete_filtered:
                    filtered_stories.append(story)
            self.feeds_df.at[index, 'stories'] = filtered_stories

        return self

    @staticmethod
    def normalize_aylien_story(story):
        """
        stories is a list of dicts, each dict is a story
        """
        # this is needed because arrow cannot serialize empty dicts
        if 'entities' in story:
            for e in story['entities']:
                if 'external_ids' in e and len(e['external_ids']) == 0:
                    del e['external_ids']
        return story

    def sample_stories_in_window(self, start, end,
                                 num_stories=20,
                                 sample_per_tick=True,
                                 overwrite_existing=False,
                                 stories_column='stories',
                                 freq='D'):
        """
        if sample_per_tick is True, return a dataframe with a time axis containing
        sampled stories at each tick
        Otherwise just directly return the stories
        """
        story_bucket_records = []
        if self.feeds_df is None:
            date_range = self.date_range(start, end, freq=freq)
            # init with UTC datetime index
            # we use only start dates thus the cutoff
            self.feeds_df = pd.DataFrame(
                columns=[stories_column],
                index=pd.DatetimeIndex(date_range[:-1], tz='UTC')
            )

        if sample_per_tick:
            date_range = self.date_range(start, end)
            start_end_tups = [(s, e) for s, e in zip(list(date_range), list(date_range)[1:])]
            for start, end in tqdm.tqdm(start_end_tups):
                # Note the polymorphic .isnull check from pandas won't work with arrays, thus try/except
                if not overwrite_existing:
                    try:
                        current_value = self.feeds_df.loc[start][stories_column]
                        # nans will be floats
                        if not type(current_value) is float:
                            if len(current_value) > 0:
                                raise ValueError
                    except ValueError:
                        logger.info(f'Already have stories for {start} to {end}')
                        continue

                logger.info(f'Getting stories for {start} to {end}')
                params = self.make_query(start, end, per_page=num_stories)
                stories = [self.normalize_aylien_story(s) for s in self.stories_endpoint(params)]
                story_bucket_records.append({'timestamp': start, stories_column: stories})
        else:
            params = self.make_query(start, end, per_page=num_stories)
            stories = [self.normalize_aylien_story(s) for s in self.stories_endpoint(params)]
            records = defaultdict(list)
            for story in stories:
                ts = self.normalize_timestamp(story['published_at'], freq)
                records[ts].append(story)
            for ts, stories in records.items():
                story_bucket_records.append({'timestamp': ts, stories_column: stories})

        # now merge the stories into self.feeds_df at the correct timestamps
        story_bucket_df = pd.DataFrame(
            story_bucket_records,
            index=pd.DatetimeIndex([r['timestamp'] for r in story_bucket_records], tz='UTC')
        )
        self.feeds_df = self.feeds_df.combine_first(story_bucket_df)

        return self

    def sample_anomaly_stories(self, start, end, num_stories=20):
        """
        get anomaly windows in range, then for each window, tell us some stories
        :return:
        """
        pass

    def summarize(self,
                  summarizer: Summarizer,
                  summarization_params=None,
                  cache_summaries=True,
                  overwrite_existing=False):

        # don't summarize if summaries already exist in df
        if "summary" in self.feeds_df.columns and not overwrite_existing:
            logger.info("summaries already exist, not summarizing")
            return self.feeds_df["summary"]

        if self.feeds_df is None:
            raise NoStoriesException(
                "Cannot summarize since no stories are cached. To cache "
                "stories, run signal.sample_stories_in_window(start, end) "
                "with cache_stories=True."
            )
        if summarization_params is None:
            summarization_params = {}
        summaries = []
        for date_idx in tqdm.tqdm(self.feeds_df.index):
            stories = self.feeds_df["stories"][date_idx]
            summary = summarizer(stories, **summarization_params)
            # summaries are always json-serializable dicts
            summaries.append(summary.to_dict())

        # side effect
        if cache_summaries:
            self.feeds_df["summary"] = summaries

        return summaries

    def add_wikimedia_pageviews_timeseries(
        self,
        wikimedia_endpoint=None,
        wikidata_client=None,
        overwrite_existing=False,
    ):
        """
        look at the params that were used to query the NewsAPI, and try to derive
        a query to the wikimedia pageviews API from that.

        For example, if there's no wikidata id in the NewsAPI query, this function should
        fail noisyly.
        """
        if not overwrite_existing and "wikimedia_pageviews" in self.timeseries_df.columns:
            logger.info("wikimedia pageviews already exist, not adding")
            return self
        try:
            wikidata_id = self.params['entity_ids'][0]
        except (KeyError, IndexError):
            try:
                wikidata_id = self.aql.split("id:")[1].split(")")[0]
                assert wikidata_id.startswith("Q")
            except Exception:
                raise WikidataIDNotFound(
                    "No Wikidata ID found in signal.params or signal.aql"
                )
        start = self.timeseries_df.index.min().to_pydatetime()
        end = self.timeseries_df.index.max().to_pydatetime()
        pageviews_df = wikidata_id_to_wikimedia_pageviews_timeseries(
            wikidata_id,
            start,
            end,
            granularity='daily',
            wikidata_client=wikidata_client,
            wikimedia_endpoint=wikimedia_endpoint,
        )
        try:
            self.timeseries_df['wikimedia_pageviews'] = pageviews_df['wikimedia_pageviews'].values
        except TypeError as e:
            logger.error(e)
            logger.warning('Retrieved wikimedia pageviews dataframe is None, not adding to signal')

        return self

    def add_yfinance_timeseries(
        self,
        ticker,
        columns=None,
        overwrite_existing=True,
        append_dates=False
    ):
        """
        Retrieve market time series data from Yahoo Finance for the instance's date range,
        and add the specified columns to self.timeseries_df.

        The date range is determined from self.start and self.end (if available),
        or from the minimum and maximum dates of self.timeseries_df's index.

        Parameters:
        - ticker (str): The stock ticker symbol to retrieve (required).
        - columns (str or list of str, optional): The column(s) to extract from the yfinance data.
            Defaults to ["Close"]. If "RSI" is included, RSI data will be retrieved.
        - overwrite_existing (bool): Whether to overwrite existing yfinance data if already present.
        - append_dates (bool):
            If True, any dates present in the yfinance data that are not already in self.timeseries_df
            will be appended (the DataFrame is reindexed to the union of dates).
            If False, only rows with dates common to both DataFrames are updated.

        Returns:
        - self: The signal instance with the new timeseries data.
        """
        # Default columns
        if columns is None:
            columns = ["Close"]
        elif isinstance(columns, str):
            columns = [columns]

        # Determine if RSI is requested based on columns
        rsi_requested = any(col.lower() == "rsi" for col in columns)

        # Determine the start and end dates
        try:
            start = str(self.start.to_pydatetime().date())
            end = str(self.end.to_pydatetime().date())
        except AttributeError:
            try:
                start = str(self.timeseries_df.index.min().to_pydatetime().date())
                end = str(self.timeseries_df.index.max().to_pydatetime().date())
            except Exception as e:
                logger.error("Could not determine start/end dates for yfinance query: " + str(e))
                return self

        # Retrieve market time series data from yfinance
        try:
            market_ts_df = retrieve_yfinance_timeseries(ticker, start, end, rsi=rsi_requested)
        except Exception as e:
            logger.error("Error retrieving yfinance timeseries: " + str(e))
            return self

        # Flatten MultiIndex columns if present
        if isinstance(market_ts_df.columns, pd.MultiIndex):
            market_ts_df.columns = [
                col[0] if col[1].upper() == ticker.upper() or col[1] == "" else col[1]
                for col in market_ts_df.columns
            ]

        # Drop RSI column if not requested explicitly
        if not rsi_requested:
            drop_cols = [col for col in market_ts_df.columns if col.lower() == "rsi"]
            if drop_cols:
                market_ts_df.drop(columns=drop_cols, inplace=True)

        # Case-insensitive matching of columns
        col_mapping = {col.lower(): col for col in market_ts_df.columns}

        # Verify each requested column exists
        for col in columns:
            if col.lower() not in col_mapping:
                logger.error(f"Expected column '{col}' not found in yfinance data.")
                return self

        # Initialize self.timeseries_df if needed
        if self.timeseries_df is None:
            self.timeseries_df = pd.DataFrame(index=market_ts_df.index)

        # Align datetime indices between DataFrames
        if self.timeseries_df.index.tz is not None and market_ts_df.index.tz is None:
            market_ts_df.index = market_ts_df.index.tz_localize(self.timeseries_df.index.tz)
        elif self.timeseries_df.index.tz is None and market_ts_df.index.tz is not None:
            market_ts_df.index = market_ts_df.index.tz_convert(None)

        # Add requested columns to self.timeseries_df
        for col in columns:
            new_col_name = col.lower()
            if new_col_name in self.timeseries_df.columns and not overwrite_existing:
                logger.info(f"Column {new_col_name} already exists, not overwriting.")
                continue
            try:
                actual_col = col_mapping[new_col_name]
                if append_dates:
                    combined_index = self.timeseries_df.index.union(market_ts_df.index)
                    self.timeseries_df = self.timeseries_df.reindex(combined_index)
                    self.timeseries_df[new_col_name] = market_ts_df[actual_col]
                else:
                    common_index = self.timeseries_df.index.intersection(market_ts_df.index)
                    self.timeseries_df.loc[common_index, new_col_name] = market_ts_df.loc[common_index, actual_col]
            except Exception as e:
                logger.error(f"Error assigning yfinance data for column '{col}': " + str(e))
                continue

        return self

    def add_wikipedia_current_events(
        self,
        overwrite_existing=False,
        feeds_column='wikipedia_current_events',
        freq='D',
        wikidata_client=None,
        wikipedia_endpoint=None,
        filter_by_wikidata_id=True
    ):
        if self.feeds_df is None:
            date_range = self.date_range(self.start, self.end, freq=freq)
            # init with UTC datetime index
            # we use only start dates thus the cutoff
            self.feeds_df = pd.DataFrame(
                columns=[feeds_column],
                index=pd.DatetimeIndex(date_range[:-1], tz='UTC')
            )
        elif not overwrite_existing:
            if "wikipedia_current_events" in self.feeds_df.columns:
                logger.info("wikipedia current events already exist, not adding")
                return self
        try:
            wikidata_id = self.params['entity_ids'][0]
        except (KeyError, IndexError):
            try:
                wikidata_id = self.aql.split("id:")[1].split(")")[0]
                assert wikidata_id.startswith("Q")
            except Exception:
                raise WikidataIDNotFound(
                    "No Wikidata ID found in signal.params or signal.aql"
                )

        start = self.start.to_pydatetime()
        end = self.end.to_pydatetime()
        event_items = wikidata_id_to_current_events(
            wikidata_id,
            start,
            end,
            filter_by_wikidata_id=filter_by_wikidata_id,
            wikipedia_endpoint=wikipedia_endpoint,
            wikidata_client=wikidata_client
        )

        date_to_events = defaultdict(list)
        for event in event_items:
            date_to_events[event['date']].append(event)

        records = []
        timestamps = []
        for date, events in sorted(date_to_events.items(), key=lambda x: x[0]):
            ts = self.normalize_timestamp(date, freq)
            timestamps.append(ts)
            records.append(
                {feeds_column: events}
            )

        # now merge the events into self.feeds_df at the correct timestamps
        events_df = pd.DataFrame(
            records,
            index=pd.DatetimeIndex(timestamps, tz='UTC')
        )
        self.feeds_df = self.feeds_df.combine_first(events_df)
        return self


class WikimediaSignal(Signal):
    """
    A Wikimedia signal uses Wikimedia-related sources,
    i.e. Wikipedia, Wikidata etc. to gather time series and text
    related to entities based on their Wikidata ID. Currently this includes:
    - Wikimedia pageviews timeseries (pageviews of Wikipedia articles)
    - Wikipedia Current Events entries
    NOTE: This signal does not require any sign-up to NewsAPI or similar,
    so it works for anyone out-of-the-box.
    """
    def __init__(
        self,
        name,
        metadata=None,
        timeseries_df=None,
        feeds_df=None,
        wikidata_id=None,
        ts_column='wikimedia_pageviews',
    ):
        super().__init__(
            name,
            metadata=metadata,
            timeseries_df=timeseries_df,
            feeds_df=feeds_df,
            ts_column=ts_column
        )
        self.wikidata_id = wikidata_id

    def to_dict(self):
        return {
            'type': type(self),
            'name': self.name,
            'metadata': self.metadata,
            'wikidata_id': self.wikidata_id,
            'timeseries_df': self.timeseries_df,
            'feeds_df': self.feeds_df,
            'ts_column': self.ts_column
        }

    @staticmethod
    def from_dict(data):
        return WikimediaSignal(
            name=data['name'],
            metadata=data['metadata'],
            wikidata_id=data['wikidata_id'],
            timeseries_df=data['timeseries_df'],
            feeds_df=data['feeds_df'],
            ts_column=data['ts_column'],
        )

    def __call__(self, start, end, freq='D', wikimedia_endpoint=None, wikidata_client=None):
        start = self.normalize_timestamp(start, freq)
        end = self.normalize_timestamp(end, freq)
        if freq not in ['D']:
            # currently we only support daily ticks on Wikimedia timeseries
            raise UnknownFrequencyArgument
        self.update(
            start=start, end=end, freq=freq,
            wikimedia_endpoint=wikimedia_endpoint,
            wikidata_client=wikidata_client
        )
        return self

    def update(self, start=None, end=None, freq='D', wikimedia_endpoint=None, wikidata_client=None):
        """
        This method should eventually update all of the data in the signal, not just
        the timeseries_df. This is a work in progress.

        Side effect: we may have other already data in the state, we want to upsert
        any new data while retaining the existing information as well
        :param start: datetime
        :param end: datetime
        """
        if end is None:
            end = self.normalize_timestamp(datetime.datetime.now(), freq)
        # if start is None, we look up to 30 days ago
        if start is None:
            default_interval = self.normalize_timestamp(
                end - datetime.timedelta(days=30),
                freq
            )
            current_end = self.timeseries_df.index.max()
            if current_end > default_interval:
                start = current_end
            else:
                start = default_interval
                logger.warning(
                    f'When updating signal, signal was either empty or the maximum, '
                    f'end date was more than 30 days ago, so we are using '
                    f'default update interval of 30 days --> {start} to {end}'
                )

        # first check if we already have this time range,
        # if so, we don't need to query again
        range_exists = \
            self.range_in_df(
                self.timeseries_df, start, end,
                freq=freq
            )
        if not range_exists:
            # update start and end to just get the data we don't have
            # we're only going to be clever about extending to the right,
            # if user wants historical data (before the data we already have),
            # everything's getting retrieved
            if self.timeseries_df is not None and start in self.timeseries_df.index:
                r = Signal.date_range(start, end, freq=freq)
                # find the first index that doesn't match
                for dt, idx_dt in zip(r, self.timeseries_df[start:].index):
                    if dt != idx_dt:
                        start = dt
                        break
            pageviews_df = self.query_wikipedia_pageviews_timeseries(
                start, end,
                wikimedia_endpoint=wikimedia_endpoint,
                wikidata_client=wikidata_client
            )
            if self.timeseries_df is None:
                self.timeseries_df = pageviews_df
            else:
                # note new values _do not_ overwrite old ones if index values
                # are the same
                self.timeseries_df = self.timeseries_df.combine_first(pageviews_df)

    def query_wikipedia_pageviews_timeseries(
        self,
        start,
        end,
        wikimedia_endpoint=None,
        wikidata_client=None,
    ):
        pageviews_df = wikidata_id_to_wikimedia_pageviews_timeseries(
            self.wikidata_id,
            start,
            end,
            granularity='daily',
            wikidata_client=wikidata_client,
            wikimedia_endpoint=wikimedia_endpoint,
        )
        return pageviews_df

    def add_wikimedia_pageviews_timeseries(
        self,
        wikimedia_endpoint=None,
        wikidata_client=None,
        overwrite_existing=False,
    ):
        """
        look at the params that were used to query the NewsAPI, and try to derive
        a query to the wikimedia pageviews API from that.

        For example, if there's no wikidata id, this function should
        fail noisyly.
        """
        if not overwrite_existing and "wikimedia_pageviews" in self.timeseries_df.columns:
            logger.info("wikimedia pageviews already exist, not adding")
            return self
        start = self.timeseries_df.index.min().to_pydatetime()
        end = self.timeseries_df.index.max().to_pydatetime()
        pageviews_df = self.query_wikipedia_pageviews_timeseries(
            start, end,
            wikimedia_endpoint=wikimedia_endpoint,
            wikidata_client=wikidata_client
        )
        self.timeseries_df['wikimedia_pageviews'] = pageviews_df['wikimedia_pageviews'].values
        return self

    def add_wikipedia_current_events(
        self,
        overwrite_existing=False,
        feeds_column='wikipedia_current_events',
        freq='D',
        filter_by_wikidata_id=True,
        wikidata_client=None,
        wikipedia_endpoint=None,
    ):
        if self.feeds_df is None:
            date_range = self.date_range(self.start, self.end, freq=freq)
            # init with UTC datetime index
            # we use only start dates thus the cutoff
            self.feeds_df = pd.DataFrame(
                columns=[feeds_column],
                index=pd.DatetimeIndex(date_range[:-1], tz='UTC')
            )
        elif not overwrite_existing:
            if "wikipedia_current_events" in self.feeds_df.columns:
                logger.info("wikipedia_current_events already exist, not adding")
                return self

        start = self.start.to_pydatetime()
        end = self.end.to_pydatetime()
        event_items = wikidata_id_to_current_events(
            self.wikidata_id,
            start,
            end,
            filter_by_wikidata_id=filter_by_wikidata_id,
            wikipedia_endpoint=wikipedia_endpoint,
            wikidata_client=wikidata_client
        )

        date_to_events = defaultdict(list)
        for event in event_items:
            date_to_events[event['date']].append(event)

        records = []
        timestamps = []
        for date, events in sorted(date_to_events.items(), key=lambda x: x[0]):
            ts = self.normalize_timestamp(date, freq)
            timestamps.append(ts)
            records.append(
                {feeds_column: events}
            )

        # now merge the events into self.feeds_df at the correct timestamps
        events_df = pd.DataFrame(
            records,
            index=pd.DatetimeIndex(timestamps, tz='UTC')
        )
        self.feeds_df = self.feeds_df.combine_first(events_df)
        return self


class AggregateSignal(Signal):
    def __init__(
        self,
        name: str,
        components: List[Signal],
        metadata: Optional[dict] = None
    ):
        super().__init__(name, metadata=metadata)
        self.components = components
        signal_names = Counter(s.name for s in components)
        for n, c in signal_names.most_common():
            if c > 1:
                logger.warning(
                    f'A signal named {n} occurs more than once '
                    'in signal components, this may make it difficult '
                    'to differentiate between signals.'
                )

    def to_dict(self):
        return {
            'type': type(self),
            'name': self.name,
            'metadata': self.metadata,
            'components': [
                c.to_dict() for c in self.components
            ]
        }

    @staticmethod
    def from_dict(data):
        return AggregateSignal(
            name=data['name'],
            metadata=data['metadata'],
            components=[
                Signal.from_dict(c) for c in data['components']
            ]
        )

    @property
    def df(self):
        start, end, freq = self.infer_index_args()
        return self.components_to_df(start, end, freq=freq)

    def components_to_df(self, start, end=None, freq='D'):
        # get the time window from all components
        realized_components = [
            c(start, end, freq=freq) for c in self.components
        ]
        # make their indexes match, then assert that everything is ok
        for c in realized_components:
            ts_df = c.timeseries_df
            ts_df['normalized_index'] = ts_df.index.floor(freq=freq)
            ts_df.set_index('normalized_index', drop=True, inplace=True)
        assert all(len(c) == len(realized_components[0]) for c in realized_components)
        return pd.concat(
            [s.timeseries_df[s.ts_column] for s in realized_components],
            axis=1
        )

    def infer_index_args(self):
        reference_index = self.components[0].timeseries_df.index
        freq = pd.infer_freq(reference_index)
        start = reference_index.min()
        end = reference_index.max()
        return start, end, freq

    def plot(self, include_aggregate=False):
        if len(self.components) and getattr(self.components[0], 'timeseries_df', None) is not None:
            start, end, freq = self.infer_index_args()
            df = self.components_to_df(start, end, freq=freq)
            agg = self(start, end, freq=freq)
            if include_aggregate:
                df = pd.concat([df, agg.timeseries_df], axis=1)
            plot = df.plot()
            return plot
        else:
            raise NotImplementedError(
                '.plot() is not supported for AggregateSignal without timeseries_df'
            )

    def __call__(self, start, end=None, freq='D'):
        # get the time window from all components
        return DataframeSignal(
            name=self.name,
            timeseries_df=self.components_to_df(start, end, freq).sum(axis=1)
        )

    def __getattr__(self, name):
        """
        Try to delegate to the underlying df if the attribute
        is not found on the signal itself.
        """
        try:
            return getattr(self.df, name)
        except AttributeError:
            raise AttributeError(
                f"type object '{type(self)}' has no attribute '{name}'"
            )


@dataclass
class UserSignal:
    """
    A signal owned by a particular user, backed by a persistent datastore
    """
    user_id: str
    signal: Signal

    def put(self, signal):
        pass


class SqliteSignalStore:
    """
    this could also be implemented as a Signal ORM,
    would probably be a better design but leaving like this
    until the interface is more solid.
    """
    def __init__(self, db_path):
        self.db_path = db_path
        self.signals = SqliteDict(
            db_path,
            tablename='signals',
            autocommit=True
        )

    def put(self, signal):
        self.signals[signal.id] = signal.to_dict()
        return signal.id

    def get(self, id):
        try:
            return Signal.from_dict(self.signals[id])
        except KeyError:
            return None

    def get_by_metadata(self, match_obj):
        """
        Naive implementation - looks at every item in the store
        """
        matches = []
        for id, signal_dict in self.signals.items():
            try:
                if match_obj.items() <= signal_dict['metadata'].items():
                    matches.append(self.get(id))
            except KeyError:
                pass

        return matches
