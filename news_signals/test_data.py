import os
import unittest
from pathlib import Path
import json

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class TestItemIndex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.timeseries = {}
        for ts in resources.glob('*.json'):
            name = ts.parts[-1].split('.')[0]
            data = json.load(open(ts))
            cls.timeseries[name] = data

        cls.resources = resources

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass


if __name__ == '__main__':
    unittest.main()
