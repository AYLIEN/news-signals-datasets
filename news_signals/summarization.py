from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Dict
import networkx as nx
import spacy


def load_spacy():
    nlp = spacy.load(
        'en_core_web_sm',
        disable=[
            "tok2vec", "tagger", "parser",
            "attribute_ruler", "lemmatizer", "ner"
        ],
    )
    nlp.enable_pipe("senter")
    return nlp


@dataclass
class Summary:
    stories: List[Dict] = None  # reference the items that were used to create this summary
    summary: str = None  # None value indicates empty summary
    metadata: Dict = None

    def to_dict(self):
        return {
            "stories": self.stories,
            "summary": self.summary,
            "metadata": self.metadata,
        }


class Summarizer(ABC):

    @abstractmethod
    def __call__(
        self,
        stories,
        title_field="title",
        text_field="body"
    ) -> dict:
        """
        Expects list of stories, each with 'title' and 'body' field.
        Returns dict with a 'summary' and optionally more fields.
        """
        raise NotImplementedError


class MultiArticleSummarizer(Summarizer):
    """
    A MultiArticleSummarizer takes a list of stories
    and returns a summary.
    Stories are dicts with 'title' and 'body' fields, some
    implementations allow renaming these fields.
    example story:
    {
        "title": "title of story",
        "body": "body of story"
    }
    """

    @classmethod
    def _dedup(cls, texts):
        seen = set()
        deduped = []
        for t in texts:
            if t not in seen:
                deduped.append(t)
                seen.add(t)
        return deduped

    @classmethod
    #  TODO: involve spacy doc
    def _sent_len(cls, sent, len_type):
        if len_type == 'chars':
            return len(sent)
        elif len_type == 'tokens':
            # TODO: use tokenizer
            return len(sent.split())
        elif len_type == 'sents':
            return 1
        else:
            raise ValueError('len_type must be in (chars|tokens|sents)')

    @classmethod
    def _truncate_text(cls, text, n_tokens):
        return " ".join(text.split()[:n_tokens])

    @classmethod
    def _sanitize_text(cls, text):
        text = " ".join(text.split())
        return text

    @classmethod
    def _sparse_page_rank_centrality(cls, X):
        S = cosine_similarity(X)
        nodes = list(range(S.shape[0]))
        graph = nx.from_numpy_array(S)
        pagerank = nx.pagerank(graph, weight='weight')
        scores = [pagerank[i] for i in nodes]
        return scores

    @classmethod
    def _sparse_centroid_centrality(cls, X):
        centroid = sparse.csr_matrix(X.sum(0))
        scores = cosine_similarity(X, centroid).reshape(-1)
        return scores

    def split_sentences(self, text):
        assert self.nlp is not None
        doc = self.nlp(text)
        sents = [s.text for s in doc.sents]
        return sents


class TfidfKeywordSummarizer(MultiArticleSummarizer):

    def __init__(self, top_k=10, vectorizer=None):
        self.top_k = top_k
        self.vectorizer = vectorizer

    """
    Concatenates story title+texts into one text and ranks words
    by the weight assigned by TfidfVectorizer.
    The vectorizer can be fitted on a large dataset beforehand, otherwise
    is dynamically built each time from given stories.
    """
    def __call__(
        self,
        stories,
        title_field="title",
        text_field="body",
        top_k=None,
    ):
        if top_k is None:
            top_k = self.top_k

        texts = [f"{s[title_field]} {s[text_field]}" for s in stories]
        text = " ".join(texts)

        try:
            if self.vectorizer is None:
                vectorizer = TfidfVectorizer(stop_words="english")
                vectorizer.fit_transform(texts)
            else:
                vectorizer = self.vectorizer
            vector = vectorizer.transform([text])
        except ValueError:
            return Summary(summary=None, metadata={"keywords": []})

        w2i = vectorizer.vocabulary_
        i2w = dict((i, w) for w, i in w2i.items())
        items = [(i2w[i], vector[0, i]) for i in range(vector.shape[1])]
        items.sort(key=lambda x: x[1], reverse=True)
        keywords = [w for w, _ in items[:top_k]]
        summary_text = " ".join(keywords)
        summary = Summary(
            summary=summary_text,
            metadata={"keywords": keywords}
        )
        return summary


class CentralTitleSummarizer(MultiArticleSummarizer):
    """
    Picks one central title from the list of stories,
    according to either textrank (pagerank) or centroid-based scoring,
    using sparse tfidf vectors.
    """

    def __init__(self, rank_method="textrank", truncate_body_tokens=250):
        assert rank_method in ("textrank", "centroid")
        self.rank_method = rank_method
        self.truncate_body_tokens = truncate_body_tokens

    def __call__(self, stories, title_field="title", text_field="body"):
        candidates = []
        for s in stories:
            title = s[title_field]
            body = s[text_field]
            # avoiding length bias
            if title.strip() != "":
                candidates.append((title, 1))
            if body.strip() != "":
                # body included to measure relevance of title better
                body = self._sanitize_text(body)
                body = self._truncate_text(body, self.truncate_body_tokens)
                candidates.append((body, 0))
        if len(candidates) == 0:
            return Summary(summary=None)

        texts, mask = zip(*candidates)
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        try:
            X = vectorizer.fit_transform(texts)
        except ValueError:
            return Summary(summary=None)

        if self.rank_method == "textrank":
            scores = self._sparse_page_rank_centrality(X)
        else:
            scores = self._sparse_centroid_centrality(X)
        items = sorted(
            zip(scores, mask, texts),
            key=lambda x: x[0],
            reverse=True
        )
        items = [x for x in items if x[1]]
        summary_text = items[0][2]
        summary = Summary(summary=summary_text)
        return summary


class CentralArticleSummarizer(MultiArticleSummarizer):
    """
    Picks one central article to represent a list of stories,
    according to either textrank (pagerank) or centroid-based scoring,
    using sparse tfidf vectors.

    Returning concatenated title + body of the article as string.

    Kept separate from CentralTitleSummarizer just to be explicit.
    """

    def __init__(self, rank_method="textrank", truncate_body_tokens=250):
        assert rank_method in ("textrank", "centroid")
        self.rank_method = rank_method
        self.truncate_body_tokens = truncate_body_tokens

    def __call__(self, stories, title_field="title", text_field="body"):
        texts = [f"{s[title_field]}\n{s[text_field]}" for s in stories]
        texts = [self._sanitize_text(t) for t in texts]
        texts = [t for t in texts if t.strip() != ""]
        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        try:
            # avoiding length bias
            texts_ = [
                self._truncate_text(t, self.truncate_body_tokens)
                for t in texts
            ]
            X = vectorizer.fit_transform(texts_)
        except ValueError:
            return Summary(summary=None)

        if self.rank_method == "textrank":
            scores = self._sparse_page_rank_centrality(X)
        else:
            scores = self._sparse_centroid_centrality(X)

        scored = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
        i = scored[0][0]
        summary_text = texts[i]
        summary = Summary(summary=summary_text)
        return summary


class CentroidExtractiveSummarizer(MultiArticleSummarizer):
    """
    Picks a diverse set of sentences which can be set to be only titles,
    sentences from the article body or both.
    Returns concatenated sentences/titles as string.
    """

    def __init__(
            self,
            n_first_sents=5,
            n_filter_sents=50,
            max_sim=0.6,
            nlp=None
    ):
        self.n_first_sents = n_first_sents
        self.n_filter_sents = n_filter_sents
        self.max_sim = max_sim
        self.min_sent_tokens = 7
        self.max_sent_tokens = 60
        if nlp is None:
            self.nlp = load_spacy()

    def _is_redundant(self, i, selected, X):
        for j in selected:
            if cosine_similarity(X[i], X[j])[0] > self.max_sim:
                return True
        return False

    def _prerank(self, X, centroid):
        scores = cosine_similarity(X, centroid).reshape(-1)
        indices = list(range(X.shape[0]))
        items = sorted(zip(indices, scores), key=lambda x: x[1], reverse=True)
        ranked_indices = [i for i, _ in items]
        return ranked_indices

    def _len_is_ok(self, sent):
        sent_len = self._sent_len(sent, "tokens")
        return self.min_sent_tokens <= sent_len <= self.max_sent_tokens

    def _get_mask(self, sents, stories, include_titles, include_bodies):
        title_set = set([s["title"] for s in stories])
        body_set = set([s["body"] for s in stories])
        mask = []
        for s in sents:
            if s in title_set and not include_titles:
                mask.append(0)
            elif s in body_set and not include_bodies:
                mask.append(0)
            else:
                mask.append(1)
        return mask

    def __call__(
        self,
        stories,
        title_field="title",
        text_field="body",
        len_type="sents",
        max_len=10,
        include_titles=False,
        include_bodies=True
    ):
        assert include_titles or include_bodies
        sents = []
        sent_to_mask = {}
        for s in stories:
            sents.append(s[title_field])
            sent_to_mask[s[title_field]] = 1 if include_titles else 0
            body_sents = self.split_sentences(s[text_field])
            body_sents = body_sents[:self.n_first_sents]
            sents += body_sents
            for s in body_sents:
                sent_to_mask[s] = (1 if include_bodies else 0)

        sents = [s for s in sents if s.strip() != ""]
        sents = self._dedup(sents)

        vectorizer = TfidfVectorizer(lowercase=True, stop_words='english')
        try:
            X = vectorizer.fit_transform(sents)
        except ValueError:
            return Summary(summary=None)

        X = sparse.csr_matrix(X)
        centroid = sparse.csr_matrix(X.sum(0))

        indices = self._prerank(X, centroid)[:self.n_filter_sents]
        remaining = [i for i in indices if sent_to_mask[sents[i]] == 1]
        selected = self.run(sents, remaining, X, centroid, max_len, len_type)
        summary_text = "\n".join([sents[i] for i in selected])
        summary = Summary(
            summary=summary_text,
            metadata={"summary_sents": [sents[i] for i in selected]}
        )
        return summary

    def run(self, sents, remaining, X, centroid, max_len, len_type):
        """
        Greedily finding set of sentences as close as possible to centroid.
        """
        selected = []
        sent_lens = [self._sent_len(s, len_type) for s in sents]
        summary_len = 0
        while len(remaining) > 0 and summary_len < max_len:
            if len(selected) > 0:
                summary_vector = sparse.vstack([X[i] for i in selected])
                summary_vector = sparse.csr_matrix(summary_vector.sum(0))
            i_to_score = {}
            for i in remaining:
                if len(selected) > 0:
                    new_x = X[i]
                    new_summary_vector = sparse.vstack([new_x, summary_vector])
                    new_summary_vector = sparse.csr_matrix(
                        summary_vector.sum(0)
                    )
                else:
                    new_summary_vector = X[i]
                score = cosine_similarity(new_summary_vector, centroid)[0, 0]
                i_to_score[i] = score

            ranked = sorted(
                i_to_score.items(), key=lambda x: x[1], reverse=True
            )
            for i, score in ranked:
                remaining.remove(i)
                if self._len_is_ok(sents[i]) is False:
                    continue
                if self._is_redundant(i, selected, X):
                    continue
                else:
                    selected.append(i)
                    summary_len += sent_lens[i]
                    break
        return selected


def get_summarizer(cls_name):
    try:
        cls = globals()[cls_name]
    except Exception:
        raise NotImplementedError("Unknown summarizer class: {}".format(cls_name))
    return cls
