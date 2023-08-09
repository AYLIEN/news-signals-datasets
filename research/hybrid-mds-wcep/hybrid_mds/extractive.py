from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
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


class MultiArticleSummarizer:
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

    def split_sentences(self, text):
        assert self.nlp is not None
        doc = self.nlp(text)
        sents = [s.text for s in doc.sents]
        return sents


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
            return None

        X = sparse.csr_matrix(X)
        centroid = sparse.csr_matrix(X.sum(0))

        indices = self._prerank(X, centroid)[:self.n_filter_sents]
        remaining = [i for i in indices if sent_to_mask[sents[i]] == 1]
        selected = self.run(sents, remaining, X, centroid, max_len, len_type)
        summary_sents = [sents[i] for i in selected]
        return summary_sents

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
