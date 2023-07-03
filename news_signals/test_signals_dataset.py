import json
import os
import unittest
import shutil
import datetime
import base64
from pathlib import Path

import arrow
import pandas as pd

from news_signals.data import datetime_to_aylien_str
from news_signals.log import create_logger
from news_signals import signals, test_signals
from news_signals import signals_dataset
from news_signals.signals_dataset import SignalsDataset


logger = create_logger(__name__)


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class MockTSEndpoint:
    def __init__(self):
        self.num_calls = 0

    def __call__(self, payload):        
        start = arrow.get(payload["published_at.start"]).datetime
        end = arrow.get(payload["published_at.end"]).datetime
        # simulate Aylien API response
        ts = [
            {'count': 10, 'published_at': datetime_to_aylien_str(dt)}
            for dt in signals.Signal.date_range(start, end)
        ]
        return ts


class MockStoriesEndPoint:    

    def __init__(self):
        self.sample_stories = json.loads((resources / "sample_stories.json").read_text())

    def __call__(self, payload):
        return self.sample_stories


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

    def __call__(
        self,
        url: str,
        params: dict={},
        headers: dict={},
    ):        
        return self.response


class TestDatasetGeneration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_dataset_dir = resources / "nasdaq100_sample_dataset"
        cls.output_dataset_dir = resources / "output_dataset_dir"
        cls.input_csv = resources / "nasdaq100.small.csv"
        cls.stories_endpoint = MockStoriesEndPoint()
        cls.ts_endpoint = MockTSEndpoint()
        if cls.output_dataset_dir.exists():
            shutil.rmtree(cls.output_dataset_dir)
            cls.output_dataset_dir.mkdir()

    @classmethod
    def tearDownClass(cls):
        if cls.output_dataset_dir.exists():
            shutil.rmtree(cls.output_dataset_dir)

    def tearDown(self):
        if self.output_dataset_dir.exists():
            shutil.rmtree(self.output_dataset_dir)
    
    def generate_sample_dataset(self):
        signals_dataset.generate_dataset(
            input=Path(self.input_csv),
            output_dataset_dir=Path(self.output_dataset_dir),
            start=datetime.datetime(2023, 1, 1),
            end=datetime.datetime(2023, 1, 4),
            id_field="Wikidata ID",
            name_field="Wikidata Label",
            delete_tmp_files=True,
            stories_endpoint=self.stories_endpoint,
            ts_endpoint=self.ts_endpoint,
            compress=False
        )

    def test_generate_dataset(self):
        self.generate_sample_dataset()

        signals_ = signals_dataset.SignalsDataset.load(
            self.output_dataset_dir
        )
        for signal in signals_.values():            
            self.assertIsInstance(signal.timeseries_df, pd.DataFrame)
            for col in ["published_at", "count"]:
                self.assertIn(col, signal.timeseries_df)

            self.assertIsInstance(signal.feeds_df, pd.DataFrame)                
            for col in ["stories"]:
                self.assertIn(col, signal.feeds_df)            
                
            # we know the stories should come from the mock endpoint
            assert all(
                len(tick) == len(self.stories_endpoint.sample_stories)
                for tick in signal.feeds_df['stories']
            )

            assert signal.params is not None
            assert signal.name is not None
            assert signal.id is not None              
 
    def test_signals_dataset_from_initialized_signals(self):
        # test that we can create a dataset from signals initialized
        # elsewhere
        start = datetime.datetime(2023, 3, 1)
        end = datetime.datetime(2023, 5, 20)
        name = 'biz-crime-usa-constrained'
        params = {
            'categories': ['ay.biz.crime'],
            'source.locations.country[]': 'US',
            'language': 'en'
        }
        signal = signals.AylienSignal(
            name=name,
            params=params
        )
  
        signals_dataset.generate_dataset(
            input=[signal],
            output_dataset_dir=Path(self.output_dataset_dir),
            start=start,
            end=end,
            stories_per_day=50,
            delete_tmp_files=True,
            stories_endpoint=self.stories_endpoint,
            ts_endpoint=self.ts_endpoint,
            compress=False
        )
        dataset = signals_dataset.SignalsDataset.load(self.output_dataset_dir)
        assert list(dataset.signals.values())[0].name == name
        difference = end - start
        days = difference.days + 1  # to include the start date
        assert len(list(dataset.signals.values())[0]) == days

    def test_signal_exists(self):
        self.generate_sample_dataset()
        signals_ = signals.Signal.load(self.output_dataset_dir)
        for s in signals_:            
            assert signals_dataset.signal_exists(s, self.output_dataset_dir)


class TestSignalsDataset(test_signals.SignalTest):

    @classmethod
    def setUpClass(cls):
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        pass
    
    def test_signals_dataset_dict_interface(self):
        """
        test that signals can be stored in a dataset
        and retrieved by index
        """
        aylien_signals = self.aylien_signals()
        dataset = SignalsDataset(aylien_signals)
        for k, signal in dataset.items():
            assert dataset[k].name == signal.name

    def test_signals_dataset_metadata(self):
        aylien_signals = self.aylien_signals()
        metadata = {
            'name': 'test_aylien_signals_dataset'
        }
        dataset = SignalsDataset(
            signals=aylien_signals,
            metadata=metadata
        )
        assert dataset.metadata['name'] == metadata['name']
    
    def test_signals_dataset_df(self):
        aylien_signals = self.aylien_signals()
        dataset = SignalsDataset(aylien_signals)
        df = dataset.df()
        # long format
        assert len(df) == len(aylien_signals) * len(aylien_signals[0])
        assert len(df.columns) == 5
        # signal names are static features replicated across all timestamps
        assert set(list(df['signal_name'])) == set([s.name for s in aylien_signals])
    
    def test_save_and_load_dataset(self):
        d1 = SignalsDataset(self.aylien_signals())
        tmp_dir = Path('/tmp/test_signals_dataset')
        d1.save(tmp_dir, compress=False)
        d2 = SignalsDataset.load(tmp_dir)
        for k in d1:
            assert d1[k].name == d2[k].name
        assert json.dumps(d1.metadata) == json.dumps(d2.metadata)
        shutil.rmtree(tmp_dir)

    def test_save_to_gcs(self):
        d1 = SignalsDataset(self.aylien_signals())
        tmp_dir = Path('/tmp/test_signals_dataset')
        fake_gcs_bucket = 'fake-path'
        
        class upload_from_filename:
            @classmethod
            def set_args(cls, args):
                cls.args = args
            def __call__(self, *args):
                self.set_args(args)

        class MockGCStorage:
            def Client(self):
                class get_bucket:
                    def __init__(self, bucket_name):
                        self.bucket_name = bucket_name
                    def blob(self, path):
                        self.path = path
                        self.upload_from_filename = upload_from_filename()
                        return self
                self.get_bucket = get_bucket 
                return self
        
        mock_storage = MockGCStorage()
        signals_dataset.storage = mock_storage
        save_path = d1.save(tmp_dir, compress=True, gcs_bucket_name=fake_gcs_bucket)
        assert upload_from_filename.args[0] == save_path
    
    def test_load_from_url(self):
        cache_dir = Path('/tmp/test_signals_dataset')
        cache_dir.mkdir(parents=True)
        fake_gdrive_path = 'https://drive.google.com/fake-path'
        basename = base64.b64encode(fake_gdrive_path.encode()).decode()

        d1 = SignalsDataset(self.aylien_signals())
        _ = d1.save(cache_dir / basename, compress=False)
        # assert that the loader thinks the dataset already exists
        d2 = SignalsDataset.load(fake_gdrive_path, cache_dir=cache_dir)
        assert len(d1) == len(d2)
        shutil.rmtree(cache_dir)

        # test loading from GCS urls
        cache_dir.mkdir(parents=True)
        fake_gcs_path = 'gs://fake-path'
        # this should break because GCS bucket download requires tar.gz
        with self.assertRaises(AssertionError):
            _ = SignalsDataset.load(fake_gcs_path, cache_dir=cache_dir)
        shutil.rmtree(cache_dir)

        class download_to_filename:
            args = None
            def __call__(self, *args):
                self.args = args

        class MockGCStorage:
            def Client(self):
                class bucket:
                    def __init__(self, bucket_name):
                        self.bucket_name = bucket_name
                    def blob(self, path):
                        self.path = path
                        self.download_to_filename = download_to_filename()
                        return self
                self.bucket = bucket 
                return self
        
        signals_dataset.storage = MockGCStorage()
        fake_gcs_path = 'gs://fake-path/dataset.tar.gz'
        basename = base64.b64encode(fake_gcs_path.encode()).decode()
        with self.assertRaises(FileNotFoundError):
            _ = SignalsDataset.load(fake_gcs_path, cache_dir=cache_dir)
            assert download_to_filename.args[0] == cache_dir / basename + '.tar.gz'

    def test_plot_dataset(self):
        dataset = SignalsDataset(self.aylien_signals())
        savedir = Path('/tmp/test_plot_dataset')
        dataset.plot(savedir=savedir)
        assert os.path.exists(savedir / f'{dataset.metadata["name"]}.png')
        shutil.rmtree(savedir)
    
    def test_corr(self):
        dataset = SignalsDataset(self.aylien_signals())
        corr = dataset.corr()
        assert corr.shape == (len(dataset), len(dataset))
    
    def test_transform_dataset_signals(self):
        '''
        pipe the dataset's signals through one or more functions
        that write data into the signal's state.
        '''
        dataset = SignalsDataset(self.aylien_signals())
        def anomaly_transform(signal):
            return signal.anomaly_signal()
        dataset.map(anomaly_transform)
        assert all('anomalies' in s.columns for s in dataset.signals.values())
