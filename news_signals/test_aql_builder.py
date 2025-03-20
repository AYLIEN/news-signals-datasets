import os
import unittest
from pathlib import Path

from news_signals.aql_builder import params_to_aql

from .log import create_logger


logger = create_logger(__name__)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))

PARAMS = [{
    "language": "en",
    "published_at.start": "2022-07-01T00:00:00.0Z",
    "published_at.end": "2022-08-06T00:00:00.0Z",
    "period": "+1DAY",
    "aql": "(categories:{{taxonomy:aylien AND score:[0.7 TO *] AND id:(ay.biz.corpgov ay.biz.corpresp ay.fin.intrade)}}) ' + "
           "'AND entities: {{prominence_score:[0.7 TO *] AND surface_forms.text: (\"Microsoft\") sort_by(overall_prominence)}}"
}]


class AQLBuilderTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pass

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass

    def test_aql_in_params(self):
        params = PARAMS[0]
        aql = params_to_aql(params)
        assert params['aql'] == aql

    def test_categories_in_schema(self):
        params = {
            'categories': ['ay.biz.corpgov', 'ay.biz.corpresp', 'ay.fin.intrade']
        }
        aql = params_to_aql(params)
        expected_aql = '(categories:{{taxonomy:aylien AND score:[0.7 TO *] AND id:(ay.biz.corpgov ay.biz.corpresp ay.fin.intrade)}})'
        assert aql == expected_aql

    def test_entities_in_schema(self):
        params = {
            'entity_ids': ['Q918'],
            'min_prominence_score': 0.7
        }
        aql = params_to_aql(params)
        assert aql == 'entities: {{prominence_score:[0.7 TO *] AND (id:Q918) sort_by(overall_prominence)}}'

    def test_entities_and_surface_forms_in_schema(self):
        params = {
            'entity_ids': ['Q918'],
            'min_prominence_score': 0.7,
            'entity_surface_forms_text': ['Twitter', 'twtr', 'twitter.com', 'Twitter Inc.']
        }
        aql = params_to_aql(params)
        assert aql == 'entities: {{prominence_score:[0.7 TO *] AND ' + \
            'surface_forms.text: ("Twitter" "twtr" "twitter.com" "Twitter Inc.") ' + \
            'AND (id:Q918) sort_by(overall_prominence)}}'

        # Test with "entity_surface_forms" (not "entity_surface_forms_text")
        params = {'entity_surface_forms': ['Twitter', 'twtr', 'twitter.com', 'Twitter Inc.']}
        aql = params_to_aql(params)
        assert aql == 'entities: {{surface_forms: ("Twitter" "twtr" "twitter.com" "Twitter Inc.")}}'
