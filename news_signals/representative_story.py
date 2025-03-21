import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from news_signals.newsapi import retrieve_stories
from abc import ABC, abstractmethod
from copy import deepcopy


class RepresentativeStoryExtractor(ABC):
    @abstractmethod
    def __call__(self, stories, k=1, **kwargs):
        raise NotImplementedError


class FirstStoryExtractor(RepresentativeStoryExtractor):
    def __call__(self, stories, k=1, **kwargs):
        return stories[:k]


class CentroidStoryExtractor(RepresentativeStoryExtractor):
    @staticmethod
    def _vectorize(stories):
        titles = [s["title"] for s in stories]
        bodies = [s["body"] for s in stories]
        all_texts = titles + bodies
        vectorizer = TfidfVectorizer(stop_words="english").fit(all_texts)
        X1 = vectorizer.transform(titles)
        X2 = vectorizer.transform(bodies)
        return X1, X2

    @staticmethod
    def _select_diverse_stories(scored, X1, X2, k, max_redundancy):
        selected = []
        for i, s1, _ in scored:
            if len(selected) >= k:
                break
            select = True
            for j, _, in selected:
                sim1 = cosine_similarity(X1[i:i + 1], X1[j:j + 1])
                sim2 = cosine_similarity(X2[i:i + 1], X2[j:j + 1])
                sim = (sim1 + sim2) / 2
                if sim > max_redundancy:
                    select = False
                    break
            if select:
                selected.append((i, s1))
        return [s for _, s in selected]

    def __call__(self, stories, k=1, max_redundancy=0.2):
        """
        Pick k stories closest to centroid observing a similarity threshold to
        already selected stories for diversity (lower=more diverse).
        Centroid/story vectors are computed for titles and bodies separately
        and the scores are the mean between the two.
        """
        stories = dedup_stories(stories)
        X1, X2 = self._vectorize(stories)
        c1 = np.asarray(X1.mean(axis=0))
        c2 = np.asarray(X2.mean(axis=0))
        scores1 = cosine_similarity(X1, c1)
        scores2 = cosine_similarity(X2, c2)
        scores = scores1 + scores2
        indices = list(range(len(stories)))
        scored = sorted(
            zip(indices, stories, scores),
            key=lambda x: x[2], reverse=True
        )
        rep_stories = self._select_diverse_stories(
            scored, X1, X2, k, max_redundancy
        )
        return rep_stories


def window_to_stories(query, start, end):
    query = deepcopy(query)
    query["published_at.start"] = start
    query["published_at.end"] = end
    stories = retrieve_stories(params=query, n_pages=1)
    return stories


# TODO: convert dt to string if necessary
def windows_to_stories(query, windows, per_page=10):
    """
    Given a template query and list of time windows, we retrieve stories
    for the query for each time window individually.
    """
    story_query = dict(
        (k, v) for k, v in query.items()
        if k not in ("period") and not k.startswith("published_at")
    )
    story_query["per_page"] = per_page
    windows_and_stories = []
    for start, end in windows:
        stories = window_to_stories(query, start, end)
        windows_and_stories.append(((start, end), stories))
    return windows_and_stories


def hash_title_and_body(story):
    text = f'{story["title"]} {story["body"]}'
    return hash(" ".join(text.split()))


# TODO: reimplement as online clusterization
def dedup_stories(stories, key_func=hash_title_and_body):
    seen = set()
    deduped = []
    for s in stories:
        key = key_func(s)
        if key not in seen:
            deduped.append(s)
            seen.add(key)
    return deduped
