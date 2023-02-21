import unittest
from news_signals import representative_story


class TestRepresentativeStories(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.first_story_baseline = representative_story.FirstStoryExtractor()
        cls.centroid_method = representative_story.CentroidStoryExtractor()
        cls.stories = [
            {"title": "apple apple", "body": "apple apple"},
            {"title": "pear pear", "body": "pear pear"},
            {"title": "apple pear", "body": "apple pear"},
            {"title": "apple apple", "body": "apple pear"},
            {"title": "pear pear", "body": "apple apple"},
        ]

    @classmethod
    def tearDownClass(cls):
        pass

    def tearDown(self):
        pass

    def test_first_stories_baseline(self):
        for k in range(1, len(self.stories)):
            rep_stories = self.first_story_baseline(self.stories, k=k)
            assert rep_stories == self.stories[:k]

    def test_centroid_method_example(self):
        # stories[2] is has most representative title and body
        rep_story = self.centroid_method(self.stories)[0]
        assert rep_story == self.stories[2]

    def test_centroid_method_k(self):
        for k in range(1, len(self.stories)):
            rep_stories = self.centroid_method(
                self.stories, k=k, max_redundancy=1.
            )
            assert len(rep_stories) == k

    def test_centroid_method_redundancy(self):
        k = len(self.stories)
        rep_stories = self.centroid_method(
            self.stories, k=k, max_redundancy=1.
        )
        assert len(rep_stories) == len(self.stories)

        rep_stories = self.centroid_method(
            self.stories, k=k, max_redundancy=0.
        )
        assert len(rep_stories) == 1


if __name__ == '__main__':
    unittest.main()
