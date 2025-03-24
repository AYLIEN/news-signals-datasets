# Sample utilities for transforming datasets
from news_signals.summarization import get_summarizer
from news_signals.signals import WikidataIDNotFound
from news_signals.log import create_logger


logger = create_logger(__name__)


def add_anomalies(
    dataset,
    overwrite_existing=False
):
    def transform(signal):
        return signal.anomaly_signal(overwrite_existing=overwrite_existing)
    dataset.map(transform)
    return dataset


def add_summaries(
    dataset,
    summarizer: str = None,
    summarization_params: dict = None,
    overwrite_existing: bool = False
):
    if summarizer is None:
        summarizer = "CentralTitleSummarizer"
    if summarization_params is None:
        summarization_params = {}

    summarizer_cls = get_summarizer(summarizer)
    summarizer = summarizer_cls()

    def transform(signal):
        signal.summarize(
            summarizer=summarizer,
            summarization_params=summarization_params,
            cache_summaries=True,
            overwrite_existing=overwrite_existing
        )
        return signal
    dataset.map(transform)
    return dataset


def add_wikimedia_pageviews(
    dataset,
    wikidata_client=None,
    wikimedia_endpoint=None,
    overwrite_existing=False
):
    def transform(signal):
        return signal.add_wikimedia_pageviews_timeseries(
            wikidata_client=wikidata_client,
            wikimedia_endpoint=wikimedia_endpoint,
            overwrite_existing=overwrite_existing
        )
    try:
        dataset.map(transform)
    except WikidataIDNotFound as e:
        logger.error(f"Could not find Wikidata ID for signal: {e}, did not apply the pageviews transformation.")
    return dataset


def add_wikipedia_current_events(
    dataset,
    wikidata_client=None,
    wikipedia_endpoint=None,
    overwrite_existing=False
):
    def transform(signal):
        return signal.add_wikipedia_current_events(
            wikidata_client=wikidata_client,
            wikipedia_endpoint=wikipedia_endpoint,
            overwrite_existing=overwrite_existing
        )
    try:
        dataset.map(transform)
    except WikidataIDNotFound as e:
        logger.error(f"Could not find Wikidata ID for signal: {e}, did not apply the pageviews transformation.")
    return dataset


REGISTRY = {
    "add_anomalies": add_anomalies,
    "add_summaries": add_summaries,
    "add_wikimedia_pageviews": add_wikimedia_pageviews,
    "add_wikipedia_current_events": add_wikimedia_pageviews
}


def get_dataset_transform(func_name):
    try:
        func = REGISTRY[func_name]
    except KeyError:
        raise NotImplementedError(f'Unknown transformation function: {func_name}')
    return func
