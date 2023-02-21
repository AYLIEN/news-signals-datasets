import json
import requests
import datetime
import urllib

import pandas as pd
from ratelimit import limits, sleep_and_retry
from wikidata.client import Client

from news_signals.log import create_logger

logger = create_logger(__name__)


# TODO: set ratelimits; but sequential requests are probably too to slow hit them


class WikidataClient:
    """
    Mainly exists to replace it with Mock version in tests.
    """    
    def __init__(self):
        self.client = Client()        
    
    def __call__(self, wikidata_id):
        entity = self.client.get(wikidata_id, load=True)
        return entity.data


class RequestsEndpoint:
    """
    Mainly exists to replace it with Mock version in tests.
    """
    def __call__(
        self,
        url: str,
        params: dict={},
        headers: dict={},
    ):
        r = requests.get(url, params=params, headers=headers)
        data = json.loads(r.text)
        return data

    
def ts_records_to_ts_df(ts_records, time_field='timestamp'):
    df = pd.DataFrame(ts_records)
    df[time_field] = pd.to_datetime(df[time_field])
    df.set_index(time_field, inplace=True)
    return df


def wikimedia_pageviews_timeseries_from_wikidata_id(
    wikidata_id: str,
    start: datetime.datetime,
    end: datetime.datetime,
    granularity: str="daily",
    language: str="en",
    wikidata_client=None,
    wikimedia_endpoint=None,
) -> pd.DataFrame:
    """
    First try to get the English Wikipedia link from Wikidata using Wikidata ID,
    then use that to get pageviews from Wikimedia API.
    """
    if wikidata_client is None:
        wikidata_client = WikidataClient()
    if wikimedia_endpoint is None:
        wikimedia_endpoint = RequestsEndpoint()

    wikipedia_link = wikipedia_link_from_wikidata_id(
        wikidata_id,
        client=wikidata_client
    )
    if wikipedia_link is None:
        logger.error(f"No Wikipedia link found for entity {wikidata_id}; page views set to None.")
        return None
        
    page_views_df = wikimedia_pageviews_timeseries_from_wikipedia_link(
        wikipedia_link,
        start,
        end,
        granularity=granularity,
        language=language,
        endpoint=wikimedia_endpoint,
    )
    return page_views_df


def wikipedia_link_from_wikidata_id(
    wikidata_id: str,
    client=None,
) -> str:
    """
    Try to return the English wikipedia page for a wikidata item
    """
    if client is None:
        client = WikidataClient()
    url = None
    try:
        entity_data = client(wikidata_id)
        url = entity_data['sitelinks']['enwiki']['url']
    except KeyError:
        logger.error(f'Error: no wikipedia url found for entity data: {entity.data}')
    except urllib.error.HTTPError as e:
        logger.error(f'Error retrieving wikidata entity: {wikidata_id}')    
    return url


def wikimedia_pageviews_timeseries_from_wikipedia_link(
    wikipedia_link: str,
    start: datetime.datetime,
    end: datetime.datetime,
    endpoint = None,
    language: str="en",
    granularity: str="daily",
    wikimedia_headers: dict={"user-agent": "news-signals-datasets"}
) -> pd.DataFrame:
    """
    Requests pageviews timeseries of a Wikipedia page for a given time range from Wikimedia API.
    """
    if endpoint is None:
        endpoint = RequestsEndpoint()

    url_date_format = "%Y%m%d00"
    assert granularity in ["daily", "monthly"]
    start = start.strftime(url_date_format)
    end = end.strftime(url_date_format)
    page_name = wikipedia_link.split("/")[-1]
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{language}.wikipedia/all-access/all-agents/{page_name}/{granularity}/{start}/{end}"
    df = None
    try:
        response = endpoint(url, headers=wikimedia_headers)        
        records = [
            {
                "wikimedia_pageviews": item["views"],
                "timestamp": datetime.datetime.strptime(item["timestamp"], url_date_format)
            }
            for item in response["items"]
        ]
        df = ts_records_to_ts_df(records)
    except KeyError:
        logger.error(response)
    return df
