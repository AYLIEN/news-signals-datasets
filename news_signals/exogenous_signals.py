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
        logger.error(f'Error: no wikipedia url found for entity data: {entity_data}')
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
    start_ = start.strftime(url_date_format)
    end_ = end.strftime(url_date_format)
    page_name = wikipedia_link.split("/")[-1]
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{language}.wikipedia/all-access/all-agents/{page_name}/{granularity}/{start_}/{end_}"
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
        date_range = pd.date_range(start=start, end=end, freq="D")
        df = df.reindex(date_range, fill_value=0)
    except KeyError:
        logger.error(response)
    return df


#########################################################################
########## TOOLS FOR SEARCHING WIKIPEDIA CURRENT EVENTS PORTAL ##########
#########################################################################


import datetime
import calendar
import collections
import arrow
import tqdm
import requests
import calendar
from bs4 import BeautifulSoup
from dataclasses import dataclass, field, asdict


MONTH_NAMES = list(calendar.month_name)[1:]


def is_valid_monthly_wcep_url(url):
    '''
    good example: https://en.wikipedia.org/wiki/Portal:Current_events/October_2003
    bad examples:
    https://en.wikipedia.org/wiki/Portal:Current_events/2005_December_29
    https://en.wikipedia.org/wiki/Portal:Current_events/Middle_East/August_2006_in_the_Middle_East
    '''
    tail = url.split('Portal:Current_events')[-1]
    if len(tail.split('/')) != 2:
        return False
    parts = url.split('/')[-1].split('_')
    if len(parts) != 2:
        return False
    month, year = parts
    return month in MONTH_NAMES and year.isnumeric()


def get_wcep_links_linking_here(wikipedia_id):
    page = wikipedia_id.replace('_', '+')
    url = f'https://en.wikipedia.org/wiki/Special:WhatLinksHere?target={page}&namespace=100&limit=100000'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    ul_element = soup.find('ul', id='mw-whatlinkshere-list')
    links = [li.a['href'] for li in ul_element.find_all('li') if li.a]
    base_url = 'https://en.wikipedia.org'
    absolute_links = [base_url + link if link.startswith('/') else link for link in links]
    monthly_wcep_links = [link for link in absolute_links if is_valid_monthly_wcep_url(link)]
    return monthly_wcep_links


def month_to_int():
    month_to_int = {}
    for i, month in enumerate(calendar.month_name):
        if i > 0:
            month_to_int[month] = i
    return month_to_int


MONTH_TO_INT = month_to_int()


@dataclass
class EventBullet:
    text: str = None
    id: str = None
    date: datetime.datetime = None
    category: str = None
    topics: list = field(default_factory=list)
    wiki_links: list = field(default_factory=list)
    references: list = field(default_factory=list)


def url_to_time(url, month_to_num):
    tail = url.split('/')[-1]
    month, year = tail.split('_')
    m = month_to_num[month]
    y = int(year)
    return datetime.datetime(year=y, month=m, day=1)


def extract_date(date_div):
    date = date_div.find('span', class_='summary')
    date = date.text.split('(')[1].split(')')[0]
    date = arrow.get(date)
    date = datetime.date(date.year, date.month, date.day)
    return date


def wiki_link_to_id(s):
    return s.split('/wiki/')[1]


def clean_event_summary(text):
    # remove sources, e.g. "Something happened. (BBC)"
    return text.split('. (')[0] + "."


def extract_event_bullets(e, date, category):
    events = []
    topic_to_child = collections.defaultdict(set)
    topic_to_parent = collections.defaultdict(set)

    def recursively_extract_event_bullets(e,
                                    date,
                                    category,
                                    prev_topics,
                                    is_root=False):
        if is_root:
            lis = e.find_all('li', recursive=False)
            result = [
                recursively_extract_event_bullets(li, date, category, [])
                for li in lis
            ]
            return result
        else:
            ul = e.find('ul')
            if ul:
                # intermediate "node", e.g. a topic an event is assigned to

                links = e.find_all('a', recursive=False)
                new_topics = []
                for link in links:
                    try:
                        new_topics.append(wiki_link_to_id(link.get('href')))
                    except:
                        pass

                lis = ul.find_all('li', recursive=False)

                for prev_topic in prev_topics:
                    for new_topic in new_topics:
                        topic_to_child[prev_topic].add(new_topic)
                        topic_to_parent[new_topic].add(prev_topic)

                topics = prev_topics + new_topics
                for li in lis:
                    recursively_extract_event_bullets(li, date, category, topics)

            else:
                # reached the "leaf", i.e. event summary
                text = e.text.split('. (')[0] + "."
                wiki_links = []
                references = []
                for link in e.find_all('a'):
                    url = link.get('href')
                    if link.get('rel') == ['nofollow']:
                        references.append(url)
                    elif url.startswith('/wiki'):
                        wiki_links.append(url)
                event = EventBullet(
                    text=text,
                    date=date,
                    category=category,
                    topics=prev_topics,
                    wiki_links=wiki_links,
                    references=references
                )
                events.append(event)

    recursively_extract_event_bullets(e, date, category, [], is_root=True)
    return events


def process_month_page_2004_to_2017(html):
    soup = BeautifulSoup(html, 'html.parser')
    days = soup.find_all('table', class_='vevent')
    events = []
    for day in days:
        date = extract_date(day)
        category = None
        desc = day.find('td', class_='description')
        for e in desc.children:
            if e.name == 'dl':
                category = e.text
            elif e.name == 'ul':
                events += extract_event_bullets(e, date, category)
    return events


def process_month_page_from_2018(html):
    soup = BeautifulSoup(html, 'html.parser')
    days = soup.find_all('div', class_='vevent')
    events = []
    for day in days:
        date = extract_date(day)
        #print('DATE:', date)
        category = None
        desc = day.find('div', class_='description')
        for e in desc.children:
            if e.name == 'div' and e.get('role') == 'heading':
                category = e.text
                #print('-'*25, 'CATEGORY:', category, '-'*25, '\n')
            elif e.name == 'ul':
                events += extract_event_bullets(e, date, category)
    return events


def wikidata_id_to_current_events(wikidata_id, start, end):

    wikipedia_link = wikipedia_link_from_wikidata_id(wikidata_id)
    wikipedia_id = wiki_link_to_id(wikipedia_link)
    wcep_links = get_wcep_links_linking_here(wikipedia_id)

    events = []
    for url in tqdm.tqdm(wcep_links):
        response = requests.get(url)
        html = response.text
        year = int(url.split('_')[-1])
        month = MONTH_TO_INT[url.split('/')[-1].split('_')[0]]
        if year < start.year or year > end.year:
            continue
        elif month < start.month or month > end.month:
            continue

        if 2004 <= year < 2018:
            events += process_month_page_2004_to_2017(html)
        elif 2018 <= year :
            events += process_month_page_from_2018(html)
    # we only keep events with link to our entity of interest
    events = [
        asdict(e) for e in events
        if any([
            urllib.parse.unquote(wiki_link).endswith(wikipedia_id)
            for wiki_link in e.wiki_links
        ])
    ]
    events.sort(key=lambda x: x['date'])
    return events
