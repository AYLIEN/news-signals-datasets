# Sample utilities for transforming datasets

from news_signals.summarization import get_summarizer


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
    dataset.map(transform)
    return dataset


def get_dataset_transform(func_name):
    try:
        func = globals()[func_name]
    except:
        raise NotImplementedError(f'Unknown transformation function: {func_name}')
    return func