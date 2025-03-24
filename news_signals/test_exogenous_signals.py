import unittest
import datetime
import json
import os
from pathlib import Path

from bs4 import BeautifulSoup
from unittest.mock import patch, Mock

from news_signals.exogenous_signals import (
    wikidata_id_to_wikipedia_link,
    wikipedia_link_to_wikimedia_pageviews_timeseries,
    wikidata_id_to_wikimedia_pageviews_timeseries,
    is_valid_monthly_wcep_url,
    clean_event_summary,
    wiki_link_to_id,
    extract_event_date,
    extract_event_bullets,
    process_daily_entry,
    process_monthly_page,
    wikidata_ids_to_labels,
    WikidataRelatedEntitiesSearcher,
    entity_name_to_wikidata_id,
)
from news_signals.log import create_logger


logger = create_logger(__name__)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
test_resources_path = path_to_file.parent / 'resources/test/wiki-current-events-portal'


class MockWikidataClient:
    def __init__(self, wikipedia_link):
        self.wikipedia_link = wikipedia_link

    def __call__(self, wikidata_id):
        response = {
            "sitelinks": {
                "enwiki": {
                    "url": self.wikipedia_link
                }
            },
        }
        return response


class MockRequestEndpoint:
    def __init__(self, response):
        self.response = response

    def __call__(
        self,
        url: str,
        params: dict={},
        headers: dict={},
        **kwargs
    ):
        return self.response


class TestWikidataTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.wikipedia_link = "https://en.wikipedia.org/wiki/Apple_Inc."
        cls.wikimedia_endpoint = MockRequestEndpoint(
            response=json.dumps(
                {
                    "items": [
                        {"views": 3, "timestamp": "2023010100"},
                        {"views": 4, "timestamp": "2023010200"},
                        {"views": 1, "timestamp": "2023010300"},
                        {"views": 2, "timestamp": "2023010400"},
                        {"views": 3, "timestamp": "2023010500"},
                    ]
                }
            )
        )
        cls.wikidata_client = MockWikidataClient(cls.wikipedia_link)
        cls.start = datetime.datetime(2023, 1, 1)
        cls.end = datetime.datetime(2023, 1, 5)

    def test_wikidata_id_to_wikimedia_pageviews_timeseries(self):
        pageviews_df = wikidata_id_to_wikimedia_pageviews_timeseries(
            "Q95",
            start=self.start,
            end=self.end,
            wikidata_client=self.wikidata_client,
            wikimedia_endpoint=self.wikimedia_endpoint,
        )
        values = list(pageviews_df["wikimedia_pageviews"].values)
        assert values == [3, 4, 1, 2, 3]

    def test_wikidata_id_to_wikipedia_link(self):
        wikipedia_link = wikidata_id_to_wikipedia_link(
            "Q95",
            client=self.wikidata_client,
        )
        self.assertEqual(wikipedia_link, self.wikipedia_link)

    def test_wikipedia_link_to_wikimedia_pageviews_timeseries(self):
        pageviews_df = wikipedia_link_to_wikimedia_pageviews_timeseries(
            self.wikipedia_link,
            start=self.start,
            end=self.end,
            endpoint=self.wikimedia_endpoint,
        )
        values = list(pageviews_df["wikimedia_pageviews"].values)
        assert values == [3, 4, 1, 2, 3]


class TestWikiCurrentEventsTools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        html_path = test_resources_path / 'example_monthly_page_jan_2023.html'
        with open(html_path) as f:
            html = f.read()
        cls.monthly_page_html = html

    def test_is_valid_monthly_wcep_url(self):
        good_urls = [
            'https://en.wikipedia.org/wiki/Portal:Current_events/October_2003',
            'https://en.wikipedia.org/wiki/Portal:Current_events/January_2023'
        ]
        bad_urls = [
            'https://en.wikipedia.org/wiki/Portal:Current_events/2005_December_29',
            'https://en.wikipedia.org/wiki/Portal:Current_events/Middle_East/August_2006_in_the_Middle_East'
        ]
        for url in good_urls:
            assert is_valid_monthly_wcep_url(url)
        for url in bad_urls:
            assert not is_valid_monthly_wcep_url(url)

    def test_wiki_link_to_id(self):
        url = 'https://en.wikipedia.org/wiki/Elon_Musk'
        wiki_id = 'Elon_Musk'
        assert wiki_link_to_id(url) == wiki_id

    def test_clean_event_summary(self):
        input_text = 'Somalia is admitted as the 8th member of the East African Community. (BBC News)'
        output_text = 'Somalia is admitted as the 8th member of the East African Community.'
        assert clean_event_summary(input_text) == output_text

    def assert_successful_event_parsing(self, events):
        references = []
        topics = []
        wiki_links = []
        for e in events:
            assert e.text is not None
            assert e.date is not None
            assert e.category is not None
            references += e.references
            topics += e.topics
            wiki_links += e.wiki_links
        assert any([len(x) > 0 for x in references])
        assert any([len(x) > 0 for x in topics])
        assert any([len(x) > 0 for x in wiki_links])

    def test_extract_event_bullets(self):
        soup = BeautifulSoup(self.monthly_page_html, 'html.parser')
        day_elements = soup.find_all('div', class_='current-events-main vevent')
        day_element = day_elements[0]
        date = extract_event_date(day_element)
        desc = day_element.find('div', class_='description')
        category = None
        events = []
        for e in desc.children:
            if e.name == 'p':
                b = e.find('b')
                if b is not None:
                    category = b.text
            elif e.name == 'ul':
                events += extract_event_bullets(e, date, category)
        assert len(events) > 0
        self.assert_successful_event_parsing(events)

    def test_process_daily_entry(self):
        soup = BeautifulSoup(self.monthly_page_html, 'html.parser')
        day_elements = soup.find_all('div', class_='current-events-main vevent')
        day_element = day_elements[0]
        events = process_daily_entry(day_element)
        self.assert_successful_event_parsing(events)

    def test_process_monthly_page(self):
        events = process_monthly_page(self.monthly_page_html)
        self.assert_successful_event_parsing(events)


class TestWikidataSearch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.entity_name = "Apple Inc."
        cls.wikidata_id = "Q312"

    @patch('news_signals.exogenous_signals.requests.get')
    def test_entity_to_wikidata_success(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {
            "search": [{"id": self.wikidata_id}]
        }
        mock_get.return_value = mock_response

        result = entity_name_to_wikidata_id(self.entity_name)
        self.assertEqual(result, self.wikidata_id)

    @patch('news_signals.exogenous_signals.requests.get')
    def test_entity_to_wikidata_not_found(self, mock_get):
        mock_response = Mock()
        mock_response.json.return_value = {"search": []}
        mock_get.return_value = mock_response

        result = entity_name_to_wikidata_id("NonExistentEntity")
        self.assertIsNone(result)

    @patch('news_signals.exogenous_signals.requests.get')
    def test_wikidata_related_entities_depth_1(self, mock_get):
        mock_response_entity = Mock()
        mock_response_entity.json.return_value = {"search": [{"id": self.wikidata_id}]}

        mock_response_related = Mock()
        mock_response_related.json.return_value = {
            "results": {
                "bindings": [
                    {"related": {"value": "http://www.wikidata.org/entity/Q95"}},
                    {"related": {"value": "http://www.wikidata.org/entity/Q2283"}}
                ]
            }
        }

        mock_response_labels = Mock()
        mock_response_labels.json.return_value = {
            "entities": {
                "Q95": {"labels": {"en": {"value": "Google"}}},
                "Q2283": {"labels": {"en": {"value": "Microsoft"}}}
            }
        }

        # Total of 3 calls:
        # 1. entity_to_wikidata
        # 2. wikidata_related_entities SPARQL query
        # 3. wikidata_to_ids label lookup
        mock_get.side_effect = [
            mock_response_entity,   # entity_to_wikidata
            mock_response_related,  # wikidata_related_entities
            mock_response_labels    # wikidata_to_ids
        ]

        expected_result = {"Q95": "Google", "Q2283": "Microsoft"}

        result = WikidataRelatedEntitiesSearcher(
            self.entity_name, depth=1, return_labels=False)

        self.assertEqual(result, expected_result)

    @patch('news_signals.exogenous_signals.requests.get')
    def test_wikidata_to_ids(self, mock_get):
        wikidata_ids = ["Q95", "Q2283"]
        mock_response = Mock()
        mock_response.json.return_value = {
            "entities": {
                "Q95": {"labels": {"en": {"value": "Google"}}},
                "Q2283": {"labels": {"en": {"value": "Microsoft"}}}
            }
        }
        mock_get.return_value = mock_response

        labels = wikidata_ids_to_labels(wikidata_ids)
        expected_labels = {"Q95": "Google", "Q2283": "Microsoft"}

        self.assertEqual(labels, expected_labels)
