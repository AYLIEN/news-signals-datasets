import os
import unittest
import shutil
import json
from pathlib import Path

from news_signals import signals
from news_signals.signals_dataset import SignalsDataset
from news_signals import dataset_transformations
from news_signals.log import create_logger


logger = create_logger(__name__)


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


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
        params: dict = {},
        headers: dict = {},
    ):
        return self.response


class TestDatasetTransformations(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sample_dataset_dir = resources / "nasdaq100_sample_dataset"
        cls.output_dataset_dir = resources / "output_dataset_dir"
        cls.dataset = SignalsDataset.load(cls.sample_dataset_dir)

    def tearDown(self):
        compressed_path = Path(f'{str(self.output_dataset_dir)}.tar.gz')
        if compressed_path.exists():
            compressed_path.unlink()
        if self.output_dataset_dir.exists():
            shutil.rmtree(self.output_dataset_dir)

    def save_and_load_dataset(self):
        dataset_path = self.dataset.save(self.output_dataset_dir)
        return SignalsDataset.load(dataset_path)

    def test_add_summaries(self):
        dataset_transformations.add_summaries(self.dataset)
        assert all('summary' in s.feeds_df.columns for s in self.dataset.signals.values())
        dataset = self.save_and_load_dataset()
        assert all('summary' in s.feeds_df.columns for s in dataset.signals.values())

    def test_add_anomalies(self):
        dataset_transformations.add_anomalies(self.dataset)
        assert all('anomalies' in s.columns for s in self.dataset.signals.values())
        dataset = self.save_and_load_dataset()
        assert all('anomalies' in s.columns for s in dataset.signals.values())

    def test_add_wikimedia_pageviews(self):
        signals_ = list(self.dataset.signals.values())
        start = signals_[0].start
        end = signals_[0].end
        url_date_format = "%Y%m%d00"
        mock_response = json.dumps({
            "items": [
                {"views": 42, "timestamp": dt.strftime(url_date_format)}
                for dt in signals.Signal.date_range(start, end)
            ]
        })
        dataset_transformations.add_wikimedia_pageviews(
            self.dataset,
            wikidata_client=MockWikidataClient("https://en.wikipedia.org/wiki/Apple_Inc."),
            wikimedia_endpoint=MockRequestsEndpoint(response=mock_response),
        )
        assert all('wikimedia_pageviews' in s.columns for s in self.dataset.signals.values())
        dataset = self.save_and_load_dataset()
        assert all('wikimedia_pageviews' in s.columns for s in dataset.signals.values())

    def test_add_wikipedia_current_events(self):
        html_path = resources / 'wiki-current-events-portal/example_monthly_page_jan_2023.html'
        example_html = html_path.read_text()
        dataset_transformations.add_wikipedia_current_events(
            self.dataset,
            wikidata_client=MockWikidataClient('https://en.wikipedia.org/wiki/COVID-19_pandemic'),
            wikipedia_endpoint=MockRequestsEndpoint(response=example_html)
        )
        assert all('wikipedia_current_events' in s.feeds_df.columns for s in self.dataset.signals.values())
        dataset = self.save_and_load_dataset()
        assert all('wikipedia_current_events' in s.feeds_df.columns for s in dataset.signals.values())
