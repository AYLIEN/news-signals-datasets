import unittest
import os
import json
from pathlib import Path
from news_signals.summarization import (
    Summary,
    TfidfKeywordSummarizer,
    CentralTitleSummarizer,
    CentralArticleSummarizer,
    CentroidExtractiveSummarizer
)
from .log import create_logger

logger = create_logger(__name__)

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
resources = Path(os.environ.get(
    'RESOURCES', path_to_file / '../resources/test'))


class TestSummary(unittest.TestCase):
    def test_init(self):
        Summary(
            summary="short summary",
            metadata={"author": "davinci"},
            stories=[{"title": "title", "body": "body"}]
        )

    def test_empty(self):
        summary = Summary()
        assert summary.summary is None
        assert summary.metadata is None
        assert summary.stories is None


class TestCentralTitleSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(resources / "tesla_stories.json") as f:
            stories = json.load(f)
        cls.stories = stories
        cls.titles = set([s["title"] for s in stories])
        cls.summarizer = CentralTitleSummarizer()

    def test_textrank(self):
        summarizer = CentralTitleSummarizer(rank_method="textrank")
        summary = summarizer(self.stories).summary
        assert summary in self.titles

    def test_centroid(self):
        summarizer = CentralTitleSummarizer(rank_method="centroid")
        summary = summarizer(self.stories).summary
        assert summary in self.titles

    def test_empty(self):
        stories = [{"title": "", "body": ""}, {"title": "", "body": ""}]
        summary = self.summarizer(stories).summary
        assert summary is None

    def test_only_stopwords(self):
        stories = [
            {"title": "a", "body": "the"}, {"title": "then", "body": "it"}
        ]
        summary = self.summarizer(stories).summary
        assert summary is None


class TestCentralArticleSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(resources / "tesla_stories.json") as f:
            stories = json.load(f)
        cls.stories = stories
        cls.titles = set([s["title"] for s in stories])
        cls.summarizer = CentralArticleSummarizer()

    def test_textrank(self):
        summarizer = CentralTitleSummarizer(rank_method="textrank")
        summary = summarizer(self.stories).summary
        assert summary is not None

    def test_centroid(self):
        summarizer = CentralTitleSummarizer(rank_method="centroid")
        summary = summarizer(self.stories).summary
        assert summary is not None

    def test_empty(self):
        stories = [{"title": "", "body": ""}, {"title": "", "body": ""}]
        summary = self.summarizer(stories).summary
        assert summary is None

    def test_only_stopwords(self):
        stories = [
            {"title": "a", "body": "the"}, {"title": "then", "body": "it"}
        ]
        summary = self.summarizer(stories).summary
        assert summary is None


class TestTfidfKeywordsSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(resources / "tesla_stories.json") as f:
            stories = json.load(f)
        cls.stories = stories
        cls.summarizer = TfidfKeywordSummarizer()

    def test_topk_in_init(self):
        for k in (2, 5, 10):
            summarizer = TfidfKeywordSummarizer(top_k=k)
            summary = summarizer(self.stories).summary
            assert len(summary.split()) == k

    def test_topk_in_call(self):
        for k in (2, 5, 10):
            summary = self.summarizer(self.stories, top_k=k)
            assert len(summary.metadata["keywords"]) == k
            assert len(summary.summary.split()) == k

    def test_empty(self):
        stories = [{"title": "", "body": ""}, {"title": "", "body": ""}]
        summary = self.summarizer(stories).summary
        assert summary is None

    def test_only_stopwords(self):
        stories = [
            {"title": "a", "body": "the"}, {"title": "then", "body": "it"}
        ]
        summary = self.summarizer(stories).summary
        assert summary is None


class TestCentroidExtractiveSummarizer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open(resources / "tesla_stories.json") as f:
            stories = json.load(f)
        cls.stories = stories
        cls.summarizer = CentroidExtractiveSummarizer()

    def test_redundancy(self):
        summarizer1 = CentroidExtractiveSummarizer(max_sim=0)
        summarizer2 = CentroidExtractiveSummarizer(max_sim=1)
        summary1 = summarizer1(self.stories).summary
        summary2 = summarizer2(self.stories).summary
        assert summary1 != summary2

    def test_length_types(self):
        len_types = ("chars", "tokens", "sents")
        limits = (100, 40, 2)
        for len_type, limit in zip(len_types, limits):
            summary = self.summarizer(
                self.stories, len_type="chars", max_len=40
            ).summary
            assert self.summarizer._sent_len(summary, len_type) <= limit

    def test_empty(self):
        stories = [{"title": "", "body": ""}, {"title": "", "body": ""}]
        summary = self.summarizer(stories).summary
        assert summary is None

    def test_only_stopwords(self):
        stories = [
            {"title": "a", "body": "the"}, {"title": "then", "body": "it"}
        ]
        summary = self.summarizer(stories).summary
        assert summary is None
