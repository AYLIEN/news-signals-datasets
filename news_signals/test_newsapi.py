import os
import unittest
from pathlib import Path

from news_signals import newsapi

from .log import create_logger


logger = create_logger(__name__)


path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class NewsapiTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass

    def test_create_newsapi_query(self):
        params = {
            'entity_surface_forms': ['Tesla'],
            'categories': ['ay.fin.reports', 'ay.impact.crime']
        }
        query = newsapi.create_newsapi_query(params)
        assert "id:(ay.fin.reports ay.impact.crime)" in query['aql']
        params = {
            'entity_surface_forms': 'Tesla',
            'categories': ['ay.fin.reports', 'ay.impact.crime']
        }
        with self.assertRaises(TypeError):
            _ = newsapi.create_newsapi_query(params)


class TestResponseValidation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.endpoints = [
            newsapi.STORIES_ENDPOINT,
            newsapi.CLUSTERS_ENDPOINT,
            newsapi.TIMESERIES_ENDPOINT
        ]
        cls.unprocessable_error_data = {"errors": [{
            'id': '...',
            'type': 'http://httpstatus.es/422',
            'title': 'Unprocessable Entity',
            'status': 422,
            'code': 'KB422',
            'detail': '...'
        }]}

    def test_successful_requests(self):
        newsapi.validate_newsapi_response(
            newsapi.STORIES_ENDPOINT, {"stories": []})
        newsapi.validate_newsapi_response(
            newsapi.CLUSTERS_ENDPOINT, {"clusters": []})
        newsapi.validate_newsapi_response(
            newsapi.TIMESERIES_ENDPOINT, {"time_series": []})

    def test_429_error(self):
        data = {"errors": [{
            'code': 'KB429',
            'detail': "You've exceeded your hits per minute rate limit (...)",
            'id': 'error429_too_many_request',
            'links': {'about': 'https://docs.aylien.com/newsapi/#rate-limits'},
            'status': 429,
            'title': 'Too Many Requests'
        }]}
        for endpoint in self.endpoints:
            with self.assertRaises(newsapi.TooManyRequestsError):
                newsapi.validate_newsapi_response(endpoint, data)

    def test_stories_error(self):
        with self.assertRaises(newsapi.StoriesEndpointError):
            newsapi.validate_newsapi_response(
                newsapi.STORIES_ENDPOINT,
                self.unprocessable_error_data
            )

    def test_clusters_error(self):
        with self.assertRaises(newsapi.ClustersEndpointError):
            newsapi.validate_newsapi_response(
                newsapi.CLUSTERS_ENDPOINT,
                self.unprocessable_error_data
            )

    def test_timeseries_error(self):
        with self.assertRaises(newsapi.TimeseriesEndpointError):
            newsapi.validate_newsapi_response(
                newsapi.TIMESERIES_ENDPOINT,
                self.unprocessable_error_data
            )
