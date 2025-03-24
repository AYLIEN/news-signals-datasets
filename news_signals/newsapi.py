import json
import requests
import os
import time
from copy import deepcopy
from ratelimit import limits, sleep_and_retry
from .log import create_logger
from .aql_builder import params_to_aql


logger = create_logger(__name__)

HEADERS = {
    'X-AYLIEN-NewsAPI-Application-ID': os.getenv('NEWSAPI_APP_ID'),
    'X-AYLIEN-NewsAPI-Application-Key': os.getenv('NEWSAPI_APP_KEY')
}
STORIES_ENDPOINT = 'https://api.aylien.com/news/stories'
CLUSTERS_ENDPOINT = 'https://api.aylien.com/news/clusters'
TRENDS_ENDPOINT = 'https://api.aylien.com/news/trends'
TIMESERIES_ENDPOINT = 'https://api.aylien.com/news/time_series'

try:
    NEWSAPI_CALLS_PER_MINUTE = int(os.getenv("NEWSAPI_CALLS_PER_MINUTE"))
except Exception:
    NEWSAPI_CALLS_PER_MINUTE = 60

CONNECTION_ERRORS = (
    ConnectionError,
    requests.exceptions.ConnectionError,
    requests.exceptions.ChunkedEncodingError
)


class TimeseriesEndpointError(Exception):
    pass


class StoriesEndpointError(Exception):
    pass


class ClustersEndpointError(Exception):
    pass


class TooManyRequestsError(Exception):
    pass


def set_headers(app_id=None, app_key=None, token=None):
    """
    set global headers for newsapi requests
    """
    if app_id is not None:
        HEADERS['X-AYLIEN-NewsAPI-Application-ID'] = app_id

    if app_key is not None:
        HEADERS['X-AYLIEN-NewsAPI-Application-Key'] = app_key

    if token is not None:
        HEADERS['Authorization'] = "Bearer {}".format(token)


@sleep_and_retry
@limits(calls=NEWSAPI_CALLS_PER_MINUTE, period=60)
def make_newsapi_request(
        endpoint,
        params,
        headers,
        trials=10,
        wait_seconds=30,
):
    for i in range(trials):
        try:
            response = requests.get(
                endpoint,
                params,
                headers=headers
            )
            data = json.loads(response.text)
            validate_newsapi_response(endpoint, data)
            return data
        except TooManyRequestsError as e:
            logger.info(
                f"Too-many-requests error (429) from endpoint: {e}. "
                f" Waiting {wait_seconds} seconds. Trial: {i}/{trials}"
            )
            time.sleep(wait_seconds)
        except CONNECTION_ERRORS as e:
            logger.info(
                f"Connection error: {e}. "
                f" Waiting {wait_seconds} seconds. Trial: {i}/{trials}"
            )
            time.sleep(wait_seconds)
        except json.decoder.JSONDecodeError:
            logger.error(f"exception from decoding this json: {response.text}")
            logger.error(f"status code: {response.status_code}, retrying")
            time.sleep(wait_seconds)
        except Exception as e:
            logger.error("uncaught exception validating request")
            raise e


def validate_newsapi_response(endpoint, data):
    if "errors" in data and len(data["errors"]) > 0:
        if data["errors"][0]["status"] == 429:
            raise TooManyRequestsError

    if endpoint.endswith("stories"):
        if "errors" in data:
            raise StoriesEndpointError(
                str(data['errors'])
            )
        elif "stories" not in data:
            # not sure if this ever happens
            raise StoriesEndpointError(
                'newsapi response does not contain '
                'any "stories" or "errors" field'
            )
    if endpoint.endswith("clusters"):
        if "errors" in data:
            raise ClustersEndpointError(
                str(data['errors'])
            )
        elif "clusters" not in data:
            # not sure if this ever happens
            raise ClustersEndpointError(
                'newsapi response does not contain '
                'any "clusters" or "errors" field'
            )
    if endpoint.endswith("time_series"):
        if "errors" in data:
            raise TimeseriesEndpointError(
                str(data['errors'])
            )
        elif "time_series" not in data:
            # not sure if this ever happens
            raise TimeseriesEndpointError(
                'newsapi response does not contain '
                'any "time_series" or "errors" field'
            )


def create_newsapi_query(params):
    template = {
        "language": "en",
        "period": "+1DAY"
    }
    aql = params_to_aql(params)
    return dict(template, **{'aql': aql})


MAX_STORIES_PER_PAGE = 10  # newsapi constant 2024-07-08


def retrieve_stories(params,
                     n_pages=1,
                     headers=HEADERS,
                     endpoint=STORIES_ENDPOINT,
                     verbose=False):
    params = deepcopy(params)
    stories = []
    cursor = '*'
    if 'num_stories' in params and params['num_stories'] > MAX_STORIES_PER_PAGE:
        if n_pages > 1:
            logger.warning(
                f"num_stories > {MAX_STORIES_PER_PAGE}, num_stories "
                f"will be ignored in favor of n_pages"
            )
        else:
            n_pages = params['num_stories'] // MAX_STORIES_PER_PAGE
            if params['num_stories'] % MAX_STORIES_PER_PAGE:
                n_pages += 1
    params['num_stories'] = MAX_STORIES_PER_PAGE
    for i in range(n_pages):
        if verbose:
            logger.info(f'page: {i}, stories: {len(stories)}')
        params['cursor'] = cursor
        data = make_newsapi_request(endpoint, params, headers)
        stories += data["stories"]
        if data.get('next_page_cursor', '*') != cursor:
            cursor = data['next_page_cursor']
        else:
            break
    return stories


def retrieve_clusters(cluster_params,
                      story_params=None,
                      get_stories=False,
                      n_cluster_pages=1,
                      n_story_pages=1,
                      headers=HEADERS,
                      clusters_endpoint=CLUSTERS_ENDPOINT,
                      stories_endpoint=STORIES_ENDPOINT):

    cluster_params = deepcopy(cluster_params)
    clusters = []
    cursor = '*'
    for i in range(n_cluster_pages):
        logger.info(f'page: {i}')
        cluster_params['cursor'] = cursor
        data = make_newsapi_request(
            clusters_endpoint, cluster_params, headers
        )
        clusters += data["clusters"]
        if data['next_page_cursor'] != cursor:
            cursor = data['next_page_cursor']
        else:
            break

    if get_stories:
        story_params = deepcopy(story_params)
        for i, c in enumerate(clusters):
            story_params['clusters'] = [c['id']]
            stories = retrieve_stories(
                params=story_params,
                headers=headers,
                endpoint=stories_endpoint,
                n_pages=n_story_pages
            )
            c['stories'] = stories
    return clusters


def retrieve_timeseries(params,
                        headers=HEADERS,
                        endpoint=TIMESERIES_ENDPOINT):
    data = make_newsapi_request(endpoint, params, headers)
    ts = data["time_series"]
    return ts
