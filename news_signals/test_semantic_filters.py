import os
from pathlib import Path

from news_signals.test_signals import SignalTest
from news_signals.semantic_filters import StoryKeywordMatchFilter
from news_signals.log import create_logger


logger = create_logger(__name__)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class TestFilterSignal(SignalTest):

    def test_filter_signal(self):
        example_signal = self.aylien_signals()[0]
        orig_stories_per_tick = [len(tick) for tick in example_signal['stories']]

        keywords = ['Million']
        filter_model = StoryKeywordMatchFilter(keywords=keywords)
        filtered_signal = example_signal.filter_stories(filter_model=filter_model)
        filtered_stories_per_tick = [len(tick) for tick in filtered_signal['stories']]
        assert (sum(orig_stories_per_tick) > sum(filtered_stories_per_tick))
        for tick_stories in filtered_signal['stories']:
            for s in tick_stories:
                assert any(kw in s['title'] for kw in keywords)

        # test don't delete filtered
        example_signal = self.aylien_signals()[0]
        filtered_signal = example_signal.filter_stories(filter_model=filter_model, delete_filtered=False)
        filtered_stories_per_tick = [len(tick) for tick in filtered_signal['stories']]
        assert (sum(orig_stories_per_tick) == sum(filtered_stories_per_tick))
