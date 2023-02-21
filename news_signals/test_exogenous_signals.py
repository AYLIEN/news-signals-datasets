import unittest
import datetime

from news_signals.exogenous_signals import (
    wikipedia_link_from_wikidata_id,
    wikimedia_pageviews_timeseries_from_wikipedia_link,
    wikimedia_pageviews_timeseries_from_wikidata_id
)
from news_signals.log import create_logger


logger = create_logger(__name__)


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


class TestExogenousSignals(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wikipedia_link = "https://en.wikipedia.org/wiki/Apple_Inc."
        cls.wikimedia_endpoint = MockRequestsEndpoint(
           response={
                "items": [
                    {"views": 3, "timestamp": "2023010100"},
                    {"views": 4, "timestamp": "2023010200"},
                    {"views": 1, "timestamp": "2023010300"},
                    {"views": 2, "timestamp": "2023010400"},
                    {"views": 3, "timestamp": "2023010500"},
                ]
            }
        )
        cls.wikidata_client = MockWikidataClient(cls.wikipedia_link)        
        cls.start = datetime.datetime(2023, 1, 1)
        cls.end = datetime.datetime(2023, 1, 5)

    def test_wikimedia_pageviews_timeseries_from_wikidata_id(self):
        pageviews_df = wikimedia_pageviews_timeseries_from_wikidata_id(
            "Q95",
            start=self.start,
            end=self.end,
            wikidata_client=self.wikidata_client,
            wikimedia_endpoint=self.wikimedia_endpoint,
        )
        values = list(pageviews_df["wikimedia_pageviews"].values)
        assert values == [3, 4, 1, 2, 3]
    
    def test_wikipedia_link_from_wikidata_id(self):
        wikipedia_link = wikipedia_link_from_wikidata_id(
            "Q95",
            client=self.wikidata_client,
        )
        self.assertEqual(wikipedia_link, self.wikipedia_link)
    
    def test_wikimedia_pageviews_timeseries_from_wikipedia_link(self):
        pageviews_df = wikimedia_pageviews_timeseries_from_wikipedia_link(
            self.wikipedia_link,
            start=self.start,
            end=self.end,
            endpoint=self.wikimedia_endpoint,
        )
        values = list(pageviews_df["wikimedia_pageviews"].values)
        assert values == [3, 4, 1, 2, 3]
