import os
import time
from copy import deepcopy
import copy
import json
import requests

HEADERS = {
    'X-AYLIEN-NewsAPI-Application-ID': os.getenv('NewsAPI_Application_ID'),
    'X-AYLIEN-NewsAPI-Application-Key': os.getenv('NewsAPI_Application_Key')
}


STORIES_ENDPOINT = 'https://api.aylien.com/news/stories'
CLUSTERS_ENDPOINT = 'https://api.aylien.com/news/clusters'
TRENDS_ENDPOINT = 'https://api.aylien.com/news/trends'
TIMESERIES_ENDPOINT = 'https://api.aylien.com/news/time_series'


def retrieve_timeseries(
        params,
        n_pages=1,
        headers=HEADERS,
        endpoint=TIMESERIES_ENDPOINT,
        sleep=None):
    params = copy.deepcopy(params)
    stories = []
    response = requests.get(
        endpoint,
        params,
        headers=headers
    )
    return json.loads(response.text)


query = {
    "title": "Tesla",
    "per_page": 5,
    "published_at.start": "NOW-1YEAR",
    "published_at.end": "NOW",
    "sort_by": "relevance",
    #     "language": "en"
}


def retrieve_stories(params,
                     n_pages=1,
                     headers=HEADERS,
                     endpoint=STORIES_ENDPOINT,
                     sleep=None):
    params = deepcopy(params)
    stories = []
    cursor = '*'
    for i in range(n_pages):
        # print(f'stories page {i+1}/{n_pages}')
        params['cursor'] = cursor
        response = requests.get(
            endpoint,
            params,
            headers=headers
        )

        data = json.loads(response.text)
        stories += data['stories']
        if data.get('next_page_cursor', '*') != cursor:
            cursor = data['next_page_cursor']
            if sleep is not None:
                time.sleep(sleep)
        else:
            break
    return stories
