import json
import requests
import datetime
import urllib
import arrow
import tqdm
import calendar
import pytz
from dataclasses import dataclass, field, asdict

import pandas as pd
from bs4 import BeautifulSoup
from wikidata.client import Client

from news_signals.log import create_logger

logger = create_logger(__name__)


# TODO: set ratelimits; but sequential requests are probably too to slow hit them


class GetRequestEndpoint:
    """
    Mainly exists to replace it with Mock version in tests.
    """
    def __call__(
        self,
        url: str,
        params: dict={},
        headers: dict={},
        load_json: bool=True
    ):
        return requests.get(url, params=params, headers=headers).text


##############################################################
########## TOOLS FOR REQUESTING WIKIMEDIA PAGEVIEWS ##########
##############################################################


class WikidataClient:
    """
    Mainly exists to replace it with Mock version in tests.
    """
    def __init__(self):
        self.client = Client()

    def __call__(self, wikidata_id):
        entity = self.client.get(wikidata_id, load=True)
        return entity.data


def wiki_pageviews_records_to_df(ts_records, time_field='timestamp'):
    df = pd.DataFrame(ts_records)
    df[time_field] = pd.to_datetime(df[time_field])
    df.set_index(time_field, inplace=True)
    return df


def wikidata_id_to_wikimedia_pageviews_timeseries(
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
        wikimedia_endpoint = GetRequestEndpoint()

    wikipedia_link = wikidata_id_to_wikipedia_link(
        wikidata_id,
        client=wikidata_client
    )
    if wikipedia_link is None:
        logger.error(f"No Wikipedia link found for entity {wikidata_id}; page views set to None.")
        return None

    page_views_df = wikipedia_link_to_wikimedia_pageviews_timeseries(
        wikipedia_link,
        start,
        end,
        granularity=granularity,
        language=language,
        endpoint=wikimedia_endpoint,
    )
    return page_views_df


def wikidata_id_to_wikipedia_link(
    wikidata_id: str,
    client=None,
) -> str:
    if client is None:
        client = WikidataClient()
    url = None
    try:
        entity_data = client(wikidata_id)
        url = entity_data['sitelinks']['enwiki']['url']
    except KeyError:
        logger.error(f'Error: no wikipedia url found for entity data: {entity_data}')
    except urllib.error.HTTPError:
        logger.error(f'Error retrieving wikidata entity: {wikidata_id}')
    return url


def wikipedia_link_to_wikimedia_pageviews_timeseries(
    wikipedia_link: str,
    start: datetime.datetime,
    end: datetime.datetime,
    endpoint=None,
    language: str="en",
    granularity: str="daily",
    wikimedia_headers: dict={"user-agent": "news-signals-datasets"}
) -> pd.DataFrame:
    """
    Requests pageviews timeseries of a Wikipedia page for a given time range from Wikimedia API.
    More info here: https://wikitech.wikimedia.org/wiki/Analytics/AQS/Pageviews
    """
    if endpoint is None:
        endpoint = GetRequestEndpoint()

    if start.tzinfo is None:
        start = pytz.utc.localize(start)
    if end.tzinfo is None:
        end = pytz.utc.localize(end)

    url_date_format = "%Y%m%d00"
    assert granularity in ["daily", "monthly"]
    start_ = start.strftime(url_date_format)
    end_ = end.strftime(url_date_format)
    page_name = wikipedia_link.split("/")[-1]
    url = f"https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/{language}.wikipedia/all-access/all-agents/{page_name}/{granularity}/{start_}/{end_}"
    df = None
    try:
        response = json.loads(endpoint(url, headers=wikimedia_headers))
        records = [
            {
                "wikimedia_pageviews": item["views"],
                # "timestamp": datetime.datetime.strptime(
                #    item["timestamp"],
                #    url_date_format
                # ).replace(tzinfo=pytz.timezone('UTC')),
                # set timezone to UTC
                "timestamp": pytz.utc.localize(datetime.datetime.strptime(item["timestamp"], url_date_format))
            }
            for item in response["items"]
        ]
        df = wiki_pageviews_records_to_df(records)
        # timezone also needs to be UTC
        date_range = pd.date_range(start=start, end=end, freq="D")
        df = df.reindex(date_range, fill_value=0)
    except KeyError:
        logger.error(response)
    return df


#########################################################################
########## TOOLS FOR SEARCHING WIKIPEDIA CURRENT EVENTS PORTAL ##########
#########################################################################

# Parsing the WCEP works with the code below as of 2023-11-30,
# but note that the page structures can change
# in the future, including the structure of past pages.
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


def get_wcep_links_linking_here(wikipedia_id, endpoint=None):
    if endpoint is None:
        endpoint = GetRequestEndpoint()
    page = wikipedia_id.replace('_', '+')
    url = f'https://en.wikipedia.org/wiki/Special:WhatLinksHere?target={page}&namespace=100&limit=100000'
    html = endpoint(url)
    soup = BeautifulSoup(html, 'html.parser')
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

    def to_dict(self):
        d = asdict(self)
        d['date'] = str(d['date'])
        return d


def url_to_time(url, month_to_num):
    tail = url.split('/')[-1]
    month, year = tail.split('_')
    m = month_to_num[month]
    y = int(year)
    return datetime.datetime(year=y, month=m, day=1)


def extract_event_date(date_div):
    date = date_div.find('span', class_='summary')
    date = date.text.split('(')[1].split(')')[0]
    date = arrow.get(date)
    date = datetime.datetime(date.year, date.month, date.day)
    return date


def wiki_link_to_id(s):
    """
    Example:
    https://en.wikipedia.org/wiki/Elon_Musk -> Elon_Musk
    """
    return s.split('/wiki/')[1]


def clean_event_summary(text):
    """
    Remove news sources, e.g.
    "Something happened. (BBC)" -> "Something happened."
    """
    return text.split('. (')[0] + "."


def extract_event_bullets(e, date, category):
    """
    Parsing out the "leave nodes" (events) in a structure as shown below
    while also keeping track of the intermediate path of topics,
    which will be passed to each EventBullet object.

    • International reactions to the 2023 Israel-Hamas war
        • Israel-Jordan relations
            • Jordan recalls its ambassador to Israel in condemnation of the ongoing war. (AFP via Zawya)
    • Afghanistan-Pakistan relations
        • Pakistan begins the mass deportation of undocumented Afghan refugees, according to Interior Minister Sarfraz Bugti. (The Guardian)
    """

    events = []

    def recursively_extract_event_bullets(
        e,
        date,
        category,
        prev_topics,
        is_root=False
    ):

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
                        topic_url = link.get('href')
                        topic_url = f'https://en.wikipedia.org{topic_url}'
                        new_topics.append(topic_url)
                    except Exception:
                        pass

                topics = prev_topics + new_topics
                for li in ul.find_all('li', recursive=False):
                    recursively_extract_event_bullets(li, date, category, topics)

            else:
                # reached the "leaf", i.e. event summary
                text = clean_event_summary(e.text)
                wiki_links = []
                references = []
                for link in e.find_all('a'):
                    url = link.get('href')
                    if link.get('rel') == ['nofollow']:
                        references.append(url)
                    elif url.startswith('/wiki'):
                        url = f'https://en.wikipedia.org{url}'
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


def process_daily_entry(day_element):
    date = extract_event_date(day_element)
    desc = day_element.find('div', class_='description')
    events = []
    category = None
    for e in desc.children:
        # TODO: sanity-check if bold always means category title
        if e.name == 'p':
            b = e.find('b')
            if b is not None:
                category = b.text
        elif e.name == 'ul':
            events += extract_event_bullets(e, date, category)
    return events


def process_monthly_page(html):
    soup = BeautifulSoup(html, 'html.parser')
    day_elements = soup.find_all('div', class_='current-events-main vevent')
    events = []
    for day_element in day_elements:
        events += process_daily_entry(day_element)
    return events


def wikidata_id_to_current_events(
    wikidata_id,
    start,
    end,
    filter_by_wikidata_id=True,
    wikidata_client=None,
    wikipedia_endpoint=None,
    linking_here_endpoint=None,
):
    if wikipedia_endpoint is None:
        wikipedia_endpoint = GetRequestEndpoint()

    wikipedia_link = wikidata_id_to_wikipedia_link(wikidata_id, wikidata_client)
    wikipedia_id = wiki_link_to_id(wikipedia_link)
    wcep_links = get_wcep_links_linking_here(wikipedia_id, endpoint=linking_here_endpoint)

    events = []
    for url in tqdm.tqdm(wcep_links):
        html = wikipedia_endpoint(url)
        year = int(url.split('_')[-1])
        month = MONTH_TO_INT[url.split('/')[-1].split('_')[0]]
        if year < start.year or year > end.year:
            continue
        elif month < start.month or month > end.month:
            continue
        events += process_monthly_page(html)

    events = [asdict(e) for e in events]
    # only keep events with link to our entity of interest
    if filter_by_wikidata_id:
        events = [
            e for e in events
            if any([
                urllib.parse.unquote(wiki_link).endswith(wikipedia_id)
                for wiki_link in e['wiki_links']
            ])
        ]
    events.sort(key=lambda x: x['date'])
    return events

#########################################################################
################# TOOLS FOR/AND BFS WIKIDATA SEARCH #####################
#########################################################################


def entity_name_to_wikidata_id(
    entity_name: str
) -> str:
    """
    Given an entity name returns its Wikidata ID.
    Uses Wikidata's wbsearchentities API.

    Eg: Input -> "Albert Einstein"
        Output -> "Q937"

    :param entity_name: Name of the entity to search for
    :return: Wikidata ID as a string, or None if not found
    """
    url = "https://www.wikidata.org/w/api.php"
    params = {
        "action": "wbsearchentities",
        "format": "json",
        "language": "en",
        "search": entity_name
    }
    response = requests.get(url, params=params)
    data = response.json()

    if "search" in data and data["search"]:
        return data["search"][0]["id"]

    print(f"No Wikidata ID found for '{entity_name}'.")
    return None


def WikidataRelatedEntitiesSearcher(
    entity_name: str,
    depth=1,
    return_labels=False,
    query=None
) -> list:
    """
    Given a Wikidata ID, performs a graph traversal to find related entities up to a specified depth.

    :param wikidata_id: The starting Wikidata ID.
    :param depth: The number of hops (levels) to traverse.
    :param return_labels: Whether to return Wikidata IDs or human-readable labels.
    :param query: Optional custom SPARQL query for retrieving related entities.
    :return: A list of related Wikidata IDs.
    """
    user_agent = "GraphTraversalBot/1.0 (your_email@example.com) Python/requests"
    endpoint_url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": user_agent}

    default_query = """
        SELECT ?related ?relatedLabel WHERE {{
            wd:{item} ?prop ?related .
            FILTER(isIRI(?related))
            FILTER EXISTS {{
                ?related wdt:P31/wdt:P279* ?type .
                VALUES ?type {{ wd:Q5 wd:Q43229 wd:Q4830453 wd:Q2424752 wd:Q431289 wd:Q732577 wd:Q11424 wd:Q571 }}
            }}
            SERVICE wikibase:label {{ bd:serviceParam wikibase:language \"[AUTO_LANGUAGE],en\" }}
        }}
    """
    if not return_labels:
        wikidata_id = entity_name_to_wikidata_id(entity_name)
        if not wikidata_id:
            print(f"No Wikidata ID found for '{entity_name}'.")
            return []
    else:
        wikidata_id = entity_name

    visited = set()
    current_level = {wikidata_id}
    all_related = set()

    for _ in range(depth):
        next_level = set()
        for item in current_level:
            current_query = query or default_query
            formatted_query = current_query.format(item=item)

            response = requests.get(endpoint_url, params={'query': formatted_query, 'format': 'json'}, headers=headers)
            result = response.json()

            for binding in result["results"]["bindings"]:
                related_uri = binding["related"]["value"]
                if related_uri.startswith("http://www.wikidata.org/entity/"):
                    related_id = related_uri.split("/")[-1]
                    if related_id not in visited:
                        next_level.add(related_id)
                        all_related.add(related_id)
            visited.add(item)
        current_level = next_level
        if not current_level:
            break

    if not return_labels:
        final_list = list(all_related)
        final_list = wikidata_ids_to_labels(final_list)
    else:
        final_list = list(all_related)

    return final_list


def wikidata_ids_to_labels(
    wikidata_ids,
    language='en'
) -> dict:
    """
    Convert Wikidata IDs to human-readable labels.

    Eg: Input -> ["Q937", "Q42"]
        Output -> {"Q937": "Albert Einstein", "Q42": "Douglas Adams"}


    :param wikidata_ids: List of Wikidata IDs (can include composite IDs).
    :param language: Language code for labels.
    :return: Dictionary mapping Wikidata IDs to labels.
    """
    user_agent = "GraphTraversalBot/1.0 (your_email@example.com) Python/requests"
    if not wikidata_ids:
        print("No Wikidata IDs provided for label lookup.")
        return {}

    base_ids = list(set(wid.split('-')[0] for wid in wikidata_ids))
    url = "https://www.wikidata.org/w/api.php"
    headers = {"User-Agent": user_agent}
    labels = {}

    max_ids = 50

    for i in range(0, len(base_ids), max_ids):
        batch_ids = base_ids[i:i + max_ids]
        ids_param = "|".join(batch_ids)

        params = {
            "action": "wbgetentities",
            "ids": ids_param,
            "format": "json",
            "props": "labels",
            "languages": language
        }

        response = requests.get(url, params=params, headers=headers)
        try:
            data = response.json()

            if "error" in data:
                print(f"Error fetching labels: {data['error']}")
                continue

            for entity_id, entity_info in data.get("entities", {}).items():
                label = entity_info.get("labels", {}).get(language, {}).get("value", "Unknown")
                labels[entity_id] = label

        except requests.exceptions.JSONDecodeError:
            print(f"Failed to decode JSON response. Raw response: {response.text}")
            continue

    return labels
