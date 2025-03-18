import streamlit as st
from pathlib import Path
import json
from PIL import Image
import os
import copy
import calendar
from collections import OrderedDict, defaultdict
from sqlitedict import SqliteDict
import uuid
import arrow
import time
import altair as alt


from streamlit_echarts import st_echarts

from news_signals.data import datetime_to_aylien_str, aylien_ts_to_df
from news_signals.newsapi import retrieve_timeseries, retrieve_stories
from news_signals.anomaly_detection import AnomalyDetector
from news_signals.representative_story import CentroidStoryExtractor
from news_signals import aql_builder

from st_aggrid import  AgGrid, DataReturnMode, GridUpdateMode, GridOptionsBuilder, JsCode

import pandas as pd
import numpy as np

import plost


# Page configuration

path_to_file = Path(os.path.dirname(os.path.abspath(__file__)))
img_path = path_to_file / 'exploding_head.png'
if img_path.exists():
    img = Image.open(img_path)
else:
    img = None  

st.set_page_config(
    page_title='Rich News Signals',
    initial_sidebar_state='collapsed',
    page_icon=img,
    layout='wide'
)

# Helper functions

def hide_menu_and_footer():
    hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    </style>
    """
    st.markdown(hide_streamlit_style, unsafe_allow_html=True)

def feed_id_to_string(feed_id):
    """
    Feed ID is a tuple: (entity_surface_form, signal_name, component_name).
    Return a string representation.
    """
    entity_sf, signal_name, component_name = feed_id
    return f'Entity: {entity_sf}\nSignal: {signal_name}\n\tComponent: {component_name}'

def render_list_param(p):
    """Render a list or None in a text area-friendly format."""
    if p is None:
        p = []
    return '\n'.join([str(u) for u in p])

def windows_of_interest_from_anomaly(
    df,
    anomaly_col='anomaly_weights',
    threshold=0.5,
    min_delta=5,
    window_range=1,
    output_format='isoformat'
):
    """
    Returns a dict of time windows where anomaly >= threshold, merging windows within `min_delta` days,
    then expanding by `window_range` days on each side.

    The return format matches your old code:
        { "2023-01-05T00:00:00Z⇨2023-01-06T00:00:00Z": 0.84, ... }
    """
    # Make sure our index is a proper datetime. We assume `df` has a "published_at" column.
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df = df.set_index('published_at').sort_index()

    if anomaly_col not in df.columns:
        return {}

    # Series of anomalies
    anomaly_series = df[anomaly_col].fillna(0.0)
    # Only keep positive anomalies
    anomaly_series = anomaly_series[anomaly_series >= threshold]

    if anomaly_series.empty:
        return {}

    # We'll track merged windows in a simple pass
    windows = []
    weights = []

    # Sort just in case
    anomaly_series = anomaly_series.sort_index()
    prev_date = None

    for date, val in anomaly_series.items():
        if prev_date is not None and (date - prev_date).days <= min_delta:
            # Merge into the last window
            windows[-1][1] = date
            weights[-1] = max(weights[-1], val)
        else:
            # Start a new window
            windows.append([date, date])
            weights.append(val)
        prev_date = date

    # Expand each window by +/- window_range days
    final_windows = []
    for (start_dt, end_dt) in windows:
        expanded_start = (arrow.get(start_dt).shift(days=-window_range))
        expanded_end = (arrow.get(end_dt).shift(days=+window_range))
        final_windows.append((expanded_start, expanded_end))

    # Convert to the old dictionary format
    #   f"{start}⇨{end}": weight
    #   with optional iso/datetime output
    window_dict = {}
    for (sd, ed), w in zip(final_windows, weights):
        if output_format == 'isoformat':
            s_str = datetime_to_aylien_str(sd.datetime)
            e_str = datetime_to_aylien_str(ed.datetime)
        else:
            s_str = sd.datetime
            e_str = ed.datetime

        key = f'{s_str}⇨{e_str}'
        window_dict[key] = float(w)

    return window_dict

def query_timeseries(query, session):
    """
    Retrieve time series data if not in cache.
    """
    key = json.dumps(query)
    if key in session['timeseries_cache']:
        return session['timeseries_cache'][key]
    else:
        data = retrieve_timeseries(query)
        session['timeseries_cache'][key] = data
        return data


def query_stories(query, session, **kwargs):
    """
    Note slight caching discrepancy -- we dont cache kwargs
    """
    key = json.dumps(query)
    if key in session['stories_cache']:
        return session['stories_cache'][key]
    else:
        data = retrieve_stories(query, **kwargs)
        session['stories_cache'][key] = data
        return data

def feed_factory(query, feed_name, session_state, metadata=None):
    """
    Creates a feed dict that includes timeseries data, anomalies, windows of interest, etc.
    """
    query['published_at.start'] = datetime_to_aylien_str(session_state['start_date'])
    query['published_at.end'] = datetime_to_aylien_str(session_state['end_date'])
    data = query_timeseries(query, session_state)

    if metadata is None:
        metadata = {}

    anomaly_detector = AnomalyDetector()
    df = aylien_ts_to_df(data) 
    if len(data) > 0 and isinstance(data[0], dict) and 'published_at' in data[0]:
        df['published_at'] = [r['published_at'] for r in data]
    else:
        df['published_at'] = pd.date_range(
            start=session_state['start_date'], 
            periods=len(df), 
            freq='D', 
            tz='UTC'
        )

    # anomaly
    df['anomaly_weights'] = anomaly_detector.history_to_anomaly_ts(df['count'])

    woi = windows_of_interest_from_anomaly(
        df, 
        anomaly_col='anomaly_weights', 
        threshold=0.5, 
        min_delta=5, 
        window_range=1, 
        output_format='isoformat'
    )

    return {
        'feed_name': feed_name,
        'query': copy.deepcopy(query),
        'df_timeseries': df,
        'aylien_timeseries': data,
        'windows_of_interest': woi,
        'metadata': metadata
    }

def format_date(d, abbr=False):
    if isinstance(d, str):
        d = arrow.get(d).datetime
    if abbr:
        return f"{d.day} {calendar.month_abbr[d.month]} {d.year}"
    else:
        return f"{d.day} {calendar.month_name[d.month]} {d.year}"

def render_feed_card(feed, idx, session_state):
    """
    Displays a 'card' showing feed metrics, anomalies, stories, etc.
    """
    with st.container():
        # The feed name is a tuple (entity_sf, signal_name, component_name)
        entity_sf, signal_name, component_name = feed["feed_name"]
        st.markdown(f'### {entity_sf} - {signal_name} - {component_name}')

        metric_col1, metric_col2, col3 = st.columns([1, 1, 6])

        # Basic stats
        total_stories = sum(feed['df_timeseries']['count'])
        metric_col1.metric('Total Stories', total_stories)
        avg_stories = round(feed['df_timeseries']['count'].mean(), 2)
        metric_col2.metric('Avg. Stories/Day', avg_stories)

        anomaly_count = len([a for a in feed['df_timeseries']['anomaly_weights'] if a > 0.])
        metric_col1.metric('Notifications', anomaly_count)

        # Anomaly score
        anomaly_df = feed['df_timeseries'][['published_at', 'anomaly_weights']]
        if not anomaly_df.empty:
            weight_now = anomaly_df['anomaly_weights'].iloc[-1]
            weight_prev = anomaly_df['anomaly_weights'].iloc[-2] if len(anomaly_df) > 1 else 0.0
        else:
            weight_now = 0.0
            weight_prev = 0.0

        weight_now = np.nan_to_num(weight_now)
        weight_prev = np.nan_to_num(weight_prev)
        metric_col2.metric(
            'Anomaly Score',
            round(weight_now, 2),
            delta=round(weight_now - weight_prev, 2),
            delta_color='inverse'
        )

        
        # Timeseries chart
        chart_key = f"volume-timeseries-echarts-{idx}"
        st_echarts(
            timeseries_lineplot(feed['aylien_timeseries'], color='rgba(255, 40, 71, 1.0)', y_axis_name='Counts', title='Volume Time Series'),
            width="100%",
            height="300px",
            key=chart_key
        )     

        # Show top story from the most recent date if we have a positive count
        if total_stories > 0:
            last_date = feed['df_timeseries']['published_at'].iloc[-1]
            notification_query = copy.deepcopy(feed['query'])
            notification_query['published_at.start'] = datetime_to_aylien_str(
                arrow.get(last_date).shift(days=-1).datetime
            )
            notification_query['published_at.end'] = datetime_to_aylien_str(
                arrow.get(last_date).shift(hours=24).datetime
            )
            notification_query['per_page'] = 30

            with col3:
                st.markdown(
                    f'##### {format_date(notification_query["published_at.start"])} - Current top stories:'
                )
                stories = query_stories(notification_query, session_state, n_pages=1)
                try:
                    representative_stories = CentroidStoryExtractor()(stories, k=3)
                except ValueError:
                    representative_stories = stories[:1]

                for s in representative_stories:
                    title_snippet = s["title"][:100] + "..." if len(s["title"]) > 100 else s["title"]
                    link = s["links"]["permalink"]
                    st.markdown(f'`{title_snippet}` [link]({link})')
        else:
            with col3:
                st.markdown('##### No recent stories')

        # Anomaly-based notifications
        with st.expander('Notification Stream'):
            if sum(anomaly_df['anomaly_weights']) == 0.0:
                st.markdown(
                    '###### No anomalies in period: '
                    f'{format_date(feed["query"]["published_at.start"])}-'
                    f'{format_date(feed["query"]["published_at.end"])}'
                )
            else:
                _, c1, c2, _ = st.columns([1, 4, 6, 1])
                anomaly_threshold = c1.slider(
                    'Anomaly Notification Threshold',
                    value=0.5,
                    min_value=0.0,
                    max_value=1.0,
                    key=f'anomaly-threshold-{idx}'
                )                            

                # Zero out anomalies below the slider threshold
                threshold_mask = anomaly_df['anomaly_weights'] < anomaly_threshold
                anomaly_df.loc[threshold_mask, 'anomaly_weights'] = 0.0
                with c2:
                    df = pd.DataFrame(anomaly_df.to_dict(orient='records'))
                    if 'published_at' in df.columns:
                        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
                        df = df.sort_values('published_at')
                    
                    chart = (
                        alt.Chart(df)
                        .mark_line(color='rgba(255, 40, 71, 1.0)', point=alt.OverlayMarkDef(color='red'))
                        .encode(
                            x=alt.X('published_at:T', axis=alt.Axis(format='%Y-%m-%d', labelAngle=-40, title=None)),
                            y=alt.Y('anomaly_weights:Q', title='Anomaly Score'),
                            tooltip=[
                                alt.Tooltip('published_at:T', title='Date', format='%Y-%m-%d'),
                                alt.Tooltip('anomaly_weights:Q', title='Anomaly Score')
                            ]
                        )
                        .properties(
                            # Centered title
                            title=alt.TitleParams(text='Anomaly Time Series', anchor='middle'),
                            width=600,  
                            height=300
                        )
                        .interactive()  
                    )

                    c2.altair_chart(chart, use_container_width=True)

                    # TODO: Check Bug, why this is not displayed?
                    # clipped_chart_key = f"anomaly-timeseries-echarts-{idx}"
                    # st_echarts(
                    #     timeseries_lineplot(
                    #         anomaly_df.to_dict(orient='records'),
                    #         y_col='anomaly_weights',
                    #         color='rgba(255, 40, 71, 1.0)',
                    #         y_axis_name='Anomaly Score',
                    #         title='Anomaly Time Series'
                    #     ),
                    #     width="100%",
                    #     height="300px",
                    #     key=clipped_chart_key
                    # )   
                    
                notification_count = len([x for x in anomaly_df['anomaly_weights'] if x > 0.])
                c1.metric('Notification Count', notification_count)
                          

                if st.button(
                    'Compute Notification Stream',
                    key=f'show-notifications-{feed_id_to_string(feed["feed_name"])}'
                ):
                    st.write(f'Retrieving {notification_count} notifications...')
                    progress_bar = st.progress(0)
                    total_points = len(anomaly_df)

                    for i, (idx2, row) in enumerate(anomaly_df.iterrows()):
                        w = row['anomaly_weights']
                        date = row['published_at']
                        if w > 0.0:
                            notification_query = copy.deepcopy(feed['query'])
                            notification_query['published_at.start'] = datetime_to_aylien_str(
                                arrow.get(date).shift(days=-1).datetime
                            )
                            notification_query['published_at.end'] = datetime_to_aylien_str(
                                arrow.get(date).shift(hours=24).datetime
                            )
                            notification_query['per_page'] = 10

                            stories = query_stories(notification_query, session_state, n_pages=1)
                            try:
                                rep_story = CentroidStoryExtractor()(stories, k=1)[0]
                            except ValueError:
                                rep_story = stories[0] if len(stories) else {}

                            st.markdown(f'#### Date: `{date}`:')
                            st.warning(
                                f'{date}: Heads up, {feed_id_to_string(feed["feed_name"])} is trending...'
                            )
                            if rep_story:
                                st.markdown(
                                    f'##### Current Top story: `{rep_story["title"][:100]}...`'
                                )
                                st.markdown(
                                    f'[Story Permalink]({rep_story["links"]["permalink"]})'
                                )
                            st.write('--------------------')

                            # Sleep artificially
                            if i < (total_points - 1):
                                time.sleep(1.5)
                        progress_bar.progress((i + 1) / total_points)

        # Top stories for entire feed
        if total_stories > 0:
            with st.expander(
                f'Top Stories: {format_date(feed["query"]["published_at.start"])} - '
                f'{format_date(feed["query"]["published_at.end"])}'
            ):
                q = copy.deepcopy(feed['query'])
                stories = query_stories(q, session_state, n_pages=1)
                try:
                    rep_stories = CentroidStoryExtractor()(stories, k=5)
                except ValueError:
                    rep_stories = stories[:5]

                for s in rep_stories:
                    st.markdown(f'###### {s["title"]}')
                    summary = s.get('summary', {}).get('sentences', [])
                    if summary:
                        st.markdown('**Summary:**')
                        for sen in summary[:2]:
                            clean_sen = " ".join(seg.strip() for seg in sen.strip().split("\n"))
                            st.markdown(f'- {clean_sen}')
                    st.write('-------------')

        # Windows of Interest
        if len(feed['windows_of_interest']) > 0:
            with st.expander('Key Time Windows'):
                st.markdown('Select time window to view key events in this feed:')
                for dates, weight in feed['windows_of_interest'].items():
                    start_date, end_date = dates.split('⇨')
                    btn_label = (
                        f'{format_date(start_date, abbr=True)}⇨{format_date(end_date, abbr=True)} '
                        f'- Conf: {weight:0.2f}'
                    )
                    btn = st.button(
                        btn_label,
                        key=f'{feed_id_to_string(feed["feed_name"])}-button-{dates}'
                    )
                    if btn:
                        window_query = copy.deepcopy(feed['query'])
                        window_query['published_at.start'] = start_date
                        window_query['published_at.end'] = end_date
                        window_query['per_page'] = 50
                        st.code(window_query)

                        window_stories = query_stories(window_query, session_state, n_pages=1)
                        try:
                            rep_win_stories = CentroidStoryExtractor()(window_stories, k=5)
                        except ValueError:
                            rep_win_stories = window_stories[:5]

                        st.markdown(
                            '##### Representative Stories for: '
                            f'{format_date(window_query["published_at.start"])} - '
                            f'{format_date(window_query["published_at.end"])}'
                        )
                        st.write('-------------')
                        for s in rep_win_stories:
                            st.markdown(f'###### {s["title"]}')
                            summary = s.get('summary', {}).get('sentences', [])
                            if summary:
                                st.markdown('**Summary:**')
                                for sen in summary[:2]:
                                    clean_sen = " ".join(seg.strip() for seg in sen.strip().split("\n"))
                                    st.markdown(f'- {clean_sen}')
                            st.write('-------------')

        # Show query
        with st.expander('Query'):
            st.code(json.dumps(feed["query"], indent=2))

def render_signal_card(signal_name, idx, session_state, renderer):
    """
    Render a rich signal with all constituent signals
    User can drill down on individual signals and understand why anomaly fired
    :param signal_name:
    :param idx:
    :param session_state:
    :return:
    """
    # TODO: signal is actually computed when this card renders, so user can dynamically
    # TODO: edit the weights
    renderer.markdown(f'### Signal: {signal_name}')
    renderer.write('------')


def get_session_state():
    """
    Build or retrieve a session-specific state dictionary from st.session_state,
    possibly with data from URL query parameters.
    """
    state = st.session_state
    query_params = st.query_params

    if not state.get('INIT', False):
        timeseries_cache = SqliteDict(
            './cache.sqlite', tablename='timeseries_cache', autocommit=True
        )
        stories_cache = SqliteDict(
            './cache.sqlite', tablename='stories_cache', autocommit=True
        )
        signals_cache = SqliteDict(
            './cache.sqlite', tablename='signals_cache', autocommit=True
        )
        signals_params_templates_cache = SqliteDict(
            './cache.sqlite', tablename='signals_params_templates_cache', autocommit=True
        )

        if 'session' in query_params:
            # Restore from the shared session cache
            session_id = query_params['session']
            session_cache = SqliteDict(
                './session_cache.sqlite', autocommit=True
            )
            user_state = session_cache[session_id]
            user_state['timeseries_cache'] = timeseries_cache
            user_state['stories_cache'] = stories_cache
            user_state['signals'] = signals_cache
            user_state['signals_params_templates'] = signals_params_templates_cache
            user_state['rich_signals'] = defaultdict(list)
        else:
            now = arrow.utcnow()
            query_template = {
                "language": "en",
                "published_at.start": datetime_to_aylien_str(now.shift(weeks=-1).datetime),
                "published_at.end": datetime_to_aylien_str(now.datetime),
                "period": "+1DAY",
            }
            user_state = {
                'mode': 'configure-signals',
                'feeds': OrderedDict(),
                'current_config_signal': 'test-signal',
                'current_config_signal_description': 'test signal description',
                'signals': signals_cache,
                'signals_params_templates': signals_params_templates_cache,
                'render_feeds': False,
                'render_volume_chart': False,
                'timeseries_cache': timeseries_cache,
                'stories_cache': stories_cache,
                'selected_feednames': None,
                'start_date': arrow.get(query_template['published_at.start']).datetime,
                'end_date': arrow.get(query_template['published_at.end']).datetime,
                'query_template': query_template,
                'rich_signals': defaultdict(list),
                'selected_signals': [],
                'digest_name': 'default digest',
                'digest_description': 'this is a news digest',
                'current_signal_config': None,
                'entity_surface_forms': []
            }

        for k, v in user_state.items():
            state[k] = v

        state['INIT'] = True

    return state


def render_signal_configuration(renderer, session_state):
   
    with renderer.expander('Configure Signal Components', expanded=True):
        params = {
            'categories': None,
            #'category_weights': None,
            'exclude_tags': None,
            'entity_surface_forms': None,
            'entities_sentiment': None,
            'include_industries': None,
            'exclude_industries': None,
            'include_text': None,
            'exclude_text': None
        }

        if params['categories'] is None:
            params['categories'] = [
                'ay.biz.hr',
                'ay.biz.fraud',
                'ay.biz.usury',
                'ay.biz.embezzle'
            ]
            # TODO: Use category weights to created weighted signals
            # params['category_weights'] = [1.0, 2.0, 3.0, 4.0]

        col1, col2, _ = renderer.columns([1, 1, 6])
        include_tags = col1.text_area(
            'Category Filters',
            value=render_list_param(params['categories']),
            height=200,
            key='config-include-categories'
        )
        include_tags = [c.strip().lower() for c in include_tags.split('\n') if len(c) > 0]
        params['categories'] = include_tags

        # weights = col2.text_area(
        #     'Category Weights',
        #     value=render_list_param(params['category_weights']),
        #     height=200,
        #     key='config-include-categories-weights'
        # )
        # weights = [float(c.strip()) for c in weights.split('\n') if len(c) > 0]

        # if len(weights) < len(include_tags):
        #     weights.extend([1.0 for _ in range(len(include_tags) - len(weights))])
        # elif len(weights) > len(include_tags):
        #     weights = weights[:len(include_tags)]
        # params['category_weights'] = weights
        # assert len(include_tags) == len(weights), 'Each category must have exactly one weight'

        sentiment_options = [None, 'negative', 'positive']
        entities_sentiment = renderer.radio(
            'Entity level sentiment filter',
            sentiment_options,
            horizontal=True,
            help='If None, everything is included; otherwise only negative/positive entity-sentiment stories.'
        )
        params['entities_sentiment'] = entities_sentiment

        # Flatten categories -> AQL
        params_template = copy.deepcopy(params)
        params_template['entity_surface_forms'] = []
        aql_bases = aql_builder.flatten_categories_to_aql(params_template)

        params_per_component = []
        for category in params_template['categories']:
            params_per_component.append(
                dict(copy.deepcopy(params_template), **{'categories': [category]})
            )

        query_base = dict(
            session_state['query_template'],
            **{
                'published_at.start': datetime_to_aylien_str(session_state['start_date']),
                'published_at.end': datetime_to_aylien_str(session_state['end_date']),
            }
        )

        query_bases = []
        if len(aql_bases) > 0:
            for aql_str in aql_bases:
                query_bases.append(dict(query_base, **{'aql': aql_str}))

            renderer.write("Sample AQL:")
            renderer.json({'aql': aql_bases[0]})
            renderer.write(f'Total components in signal: {len(aql_bases)}')

            # weight_df = pd.DataFrame(
            #     zip(params_template['categories'], weights),
            #     columns=['component', 'weight']
            # )
            # cols = renderer.columns(2)
            # with cols[0]:
            #     plost.bar_chart(
            #         data=weight_df,
            #         bar='component',
            #         value='weight',
            #         direction='horizontal'
            #     )

        current_signal_name = renderer.text_input(
            'Signal Name:',
            session_state['current_config_signal'],
            key='current-signal-name'
        )
        current_signal_description = renderer.text_input(
            'Add a short description of your signal',
            session_state['current_config_signal_description'],
            key='current-signal-description'
        )
        session_state['current_config_signal'] = current_signal_name
        session_state['current_config_signal_description'] = current_signal_description

        if renderer.button('Initialize Rich Signal'):
            if len(current_signal_name) == 0 or len(current_signal_description) == 0:
                renderer.error('You must add a name and description for your signal component')
            else:
                for qb, pt  in zip(query_bases, params_per_component):
                    component_name = f'{pt["categories"][0]}'
                    signal_template = copy.deepcopy(qb)
                    session_state['signals'][component_name] = signal_template
                    session_state['signals_params_templates'][component_name] = copy.deepcopy(pt)
                    session_state['rich_signals'][current_signal_name].append(
                        {'component_name': component_name}
                    )
                    renderer.success(f'Initialized signal component: {component_name}')

        renderer.write('----------')


def render_facet_configuration(renderer, session_state):
    """
    Configure the entity facets that will be combined with the signals.
    """
    renderer.write('### Add Entity Facets')
    entity_surface_forms = renderer.text_area(
        'Entity Surface Forms',
        value=render_list_param(session_state['entity_surface_forms']),
        height=100,
        key='entity-surface-forms-text-area'
    )
    session_state['entity_surface_forms'] = [
        sf for sf in entity_surface_forms.strip().split('\n') if len(sf) > 0
    ]

def timeseries_lineplot(time_series_data, color='rgba(255, 40, 71, 1.0)', y_col='count', y_axis_name='Count', title='Volume Time Series'):
    """
    Produce an ECharts option dict for a basic line plot. 
    time_series_data is a list of dicts with keys: published_at, count, ...
    """
    if not time_series_data:
        return {}

    # Convert to DataFrame
    df = pd.DataFrame(time_series_data)
    if 'published_at' in df.columns:
        df['published_at'] = pd.to_datetime(df['published_at'], utc=True)
        df = df.sort_values('published_at')

    # Build ECharts series data
    series_data = []
    for _, row in df.iterrows():
        dt = row['published_at'].isoformat() if hasattr(row['published_at'], 'isoformat') else str(row['published_at'])
        val = row.get(y_col, 0)
        series_data.append([dt, val])

    options = {
        'xAxis': {
            'type': 'category',
            'data': [sd[0] for sd in series_data],
        },
        'yAxis': {'type': 'value', 'name': y_axis_name},
        'series': [{
            'data': [sd[1] for sd in series_data],
            'type': 'line',
            'smooth': True,
            'color': color
        }],
        'tooltip': {
            'trigger': 'axis'
        },
        'title': {
            "text": title,
            "left": "center",   
            "top":  10          
        },
    }

    return options


def render_main_area(session_state):
    """
    Main UI flow
    """
    renderer = st

    if session_state['mode'] == 'configure-signals':
        # ---- Step 1: Configure signals ----
        render_signal_configuration(renderer, session_state)

        if len(session_state['signals']):
            selected_signals = renderer.multiselect(
                'Select rich signals to use in your digest',
                options=list(session_state['rich_signals'].keys()),
                default=list(session_state['rich_signals'].keys()),
                key='select-signals-multiselect',
                help='Choose the signals that will be used to generate your digest'
            )
            session_state['selected_signals'] = selected_signals
            render_facet_configuration(renderer, session_state)

            num_signals = len(session_state['selected_signals'])
            num_entities = len(session_state['entity_surface_forms'])
            renderer.write(f'{num_signals} signals selected')
            renderer.write(f'{num_entities} entities added')

            digests_enabled = (num_signals * num_entities) > 0

            if digests_enabled:
                renderer.write(f'Your digest will be a {num_entities} x {num_signals} matrix')
                # Global date range
                dates = renderer.date_input(
                    'Select Date Range',
                    value=(session_state['start_date'], session_state['end_date'])
                )
                if len(dates) == 2:
                    session_state['start_date'] = dates[0]
                    session_state['end_date'] = dates[1]
            else:
                renderer.write('Add at least 1 signal and 1 entity to generate a digest')

            # ---- Step 2: Instantiate rich signals as feed objects ----
            if renderer.button('Instantiate Rich Signals', key='digest_query', disabled=not digests_enabled):
                # num_feeds = num_signals * num_entities
                progress = renderer.progress(0)
                completed = 0

                for signal_name in session_state['selected_signals']:
                    for signal_component in session_state['rich_signals'][signal_name]:
                        component_name = signal_component['component_name']
                        basefeed_query = session_state['signals'][component_name]
                        feed_template = session_state['signals_params_templates'][component_name]

                        for entity_sf in session_state['entity_surface_forms']:
                            # Build an AQL that includes the entity surface form
                            entities_sentiment = feed_template.get('entities_sentiment', None)

                            aql = aql_builder.params_to_aql(
                                dict(
                                    feed_template,
                                    **{
                                        'entity_surface_forms': [entity_sf],
                                        'entities_sentiment': entities_sentiment
                                    }
                                )
                            )
                            query = copy.deepcopy(basefeed_query)
                            query['aql'] = aql

                            feed_name_tuple = (entity_sf, signal_name, component_name)
                            try:
                                session_state['feeds'][feed_name_tuple] = feed_factory(
                                    query, feed_name_tuple, session_state
                                )
                            except KeyError:
                                renderer.error(
                                    f'Could not get timeseries for {feed_id_to_string(feed_name_tuple)}, '
                                    'continuing...'
                                )
                    completed += 1
                    progress.progress(completed / num_signals)

                renderer.success(f'Generated {len(session_state["feeds"])} feeds!')
            renderer.markdown('----')

            # ---- Step 3: Optionally render signals table ----
            if len(session_state['feeds']) > 0:
                renderer.write(f'{len(session_state["feeds"])} feeds configured')
                if renderer.button('Render Signals View'):
                    session_state['mode'] = 'show-feeds'
                    st.rerun()

    else:
        # ---- Digest or feed rendering mode ----

        # Sidebar: Deeplink config
        st.sidebar.title('Create a Deeplink (optional)')
        st.sidebar.write('Use deeplinks to save and share your dashboards.')
        digest_name = st.sidebar.text_input('Digest Name', value=session_state['digest_name'])
        digest_description = st.sidebar.text_input('Digest Description', value=session_state['digest_description'])
        session_state['digest_name'] = digest_name
        session_state['digest_description'] = digest_description

        if st.sidebar.button('Generate Deeplink', key='generate-deeplink'):
            session_id = str(uuid.uuid4())
            session_cache = SqliteDict('./session_cache.sqlite', autocommit=True)

            session = {
                'digest_name': session_state['digest_name'],
                'digest_description': session_state['digest_description'],
                'mode': 'show-feeds',
                'feeds': session_state['feeds'],
                'selected_signals': session_state['selected_signals'],
                'selected_feednames': session_state['selected_feednames'],
                'render_feeds': session_state['render_feeds'],
                'render_volume_chart': session_state['render_volume_chart'],
                'start_date': session_state['start_date'],
                'end_date': session_state['end_date'],
                'query_template': session_state['query_template']
            }

            session_cache[session_id] = copy.deepcopy(session)
            # Update the URL with the new query params
            st.experimental_set_query_params(session=session_id, digest_name=session_state['digest_name'])

            st.info('URL updated with deeplink to this state.')

        # Show or hide feed data
        ad = AnomalyDetector()
        # TODO: render each rich signal separately
        # TODO: user can annotate expected anomaly dates on this signal
        # TODO: annotation will facilitate automatic and/or manual signal optimization
        for signal_idx, signal_name in enumerate(session_state['rich_signals']):
            render_signal_card(signal_name, signal_idx, session_state, st)

        feed_records = []
        for f in session_state['feeds'].values():
            total_stories = sum(f['df_timeseries']['count'])
            avg_vol = round(f['df_timeseries']['count'].mean(), 2)
            anomaly_score = round(
                ad.anomaly_weight(
                    f['df_timeseries']['count'].iloc[:-1],
                    f['df_timeseries']['count'].iloc[-1]
                ),
                2
            )
            anomaly_count = len([v for v in f['df_timeseries']['anomaly_weights'] if v > 0.])
            record = {
                'feed name': f['feed_name'],
                'total stories': total_stories,
                'average volume': avg_vol,
                'current anomaly weight': anomaly_score,
                'total anomaly count': anomaly_count
            }
            feed_records.append(record)

        feed_df = pd.DataFrame.from_records(feed_records)

        # Build AgGrid
        gb = GridOptionsBuilder.from_dataframe(feed_df)       

        gb.configure_selection(
            selection_mode="multiple",
            use_checkbox=True
        )
        gb.configure_column(
            'feed name',
            headerCheckboxSelection=True,
            headerCheckboxSelectionFilteredOnly=True
        )

        # RAG styling for anomaly weight
        cellstyle_jscode = JsCode("""
        function(params) {
            if (params.value <= 0.) {
                return {
                    'color': 'black',
                    'backgroundColor': 'green'
                }
            } else if (0 < params.value <= 0.5) {
                return {
                    'color': 'black',
                    'backgroundColor': 'yellow'
                }
            } else {
                return {
                    'color': 'black',
                    'backgroundColor': 'red'
                }
            }
        };
        """)
        gb.configure_column("current anomaly weight", cellStyle=cellstyle_jscode)

        gridOptions = gb.build()
        st.write('Sort, filter, and select from your signals, then drill down using signal cards:')
        ag_data = AgGrid(
            feed_df,
            gridOptions=gridOptions,
            key='grid1',
            update_mode=GridUpdateMode.SELECTION_CHANGED | GridUpdateMode.SORTING_CHANGED | GridUpdateMode.FILTERING_CHANGED,
            data_return_mode=DataReturnMode.FILTERED_AND_SORTED,
            fit_columns_on_grid_load=True,
            allow_unsafe_jscode=True,
        )
        selected_rows = ag_data.get('selected_rows')
        if selected_rows is not None:
            selected_rows_records = selected_rows.to_dict(orient='records')
        else:
            selected_rows_records = []

        selected_ = [r['feed name'] for r in selected_rows_records]

        selected_feeds = []
        data_records = ag_data['data'].to_dict(orient='records')
        for r in data_records:
            if r['feed name'] in selected_:
                selected_feeds.append(r['feed name'])
        session_state['selected_feednames'] = selected_feeds

        if len(selected_feeds):
            col1, col2, _ = st.columns([1, 1, 6])
            col1.metric('Selected Feeds', len(selected_feeds))
            col2.metric('Total Feeds', len(session_state['feeds']))

            # Toggle RAG chart
            if not session_state.get('render_rag_chart', False):
                if st.button('Show RAG Chart', key='rag-chart-button'):
                    session_state['render_rag_chart'] = True
                    st.rerun()
            else:
                if st.button('Hide RAG Chart', key='rag-chart-button'):
                    session_state['render_rag_chart'] = False
                    st.rerun()

            if session_state.get('render_rag_chart', False):
                rag_metric = 'current anomaly weight'
                signals_by_facet = defaultdict(dict)
                for f in feed_df.to_dict(orient='records'):
                    name = feed_id_to_string(f['feed name'])
                    signal_str = 'Signal: '
                    signal_start = name.index(signal_str)
                    facet = name[7:signal_start - 1]
                    signal = name[signal_start + len(signal_str):]
                    metric = f[rag_metric]
                    signals_by_facet[facet][signal] = metric

                all_signals = set()
                facet_records = []
                for facet, s_dict in signals_by_facet.items():
                    facet_records.append(dict({'Facet': facet}, **s_dict))
                    all_signals.update(list(s_dict.keys()))

                facet_df = pd.DataFrame.from_records(facet_records)
                facet_gb = GridOptionsBuilder.from_dataframe(facet_df)

                rag_js = JsCode("""
                function(params) {
                    if (params.value <= 0.) {
                        return {
                            'color': 'black',
                            'backgroundColor': 'green'
                        }
                    } else if (0 < params.value <= 0.5) {
                        return {
                            'color': 'black',
                            'backgroundColor': 'yellow'
                        }
                    } else {
                        return {
                            'color': 'black',
                            'backgroundColor': 'red'
                        }
                    }
                };
                """)

                for col in all_signals:
                    facet_gb.configure_column(col, cellStyle=rag_js)
                facet_grid_options = facet_gb.build()

                _ = AgGrid(
                    facet_df,
                    gridOptions=facet_grid_options,
                    key='grid2',
                    allow_unsafe_jscode=True,
                )

            # Toggle Volume Chart
            if not session_state['render_volume_chart']:
                if st.button('Show Volume Chart', key='volume-chart-button'):
                    session_state['render_volume_chart'] = True
                    st.rerun()
            else:
                if st.button('Hide Volume Chart', key='volume-chart-button'):
                    session_state['render_volume_chart'] = False
                    st.rerun()

            if session_state.get('render_volume_chart', False):

                selected_feed_keys = set(tuple(feed) for feed in session_state['selected_feednames'])

                # Filter session_state['feeds'] to only include feeds that are selected.
                filtered_feeds = {
                    feed_id: feed_obj
                    for feed_id, feed_obj in session_state["feeds"].items()
                    if feed_id in selected_feed_keys
                }
                all_ts = pd.concat(
                    [
                        feed_obj['df_timeseries']
                        .rename(columns={'count': f'{feed_id_to_string(feed_obj["feed_name"])}-count'})
                        [f'{feed_id_to_string(feed_obj["feed_name"])}-count']
                        for feed_obj in filtered_feeds.values()
                    ],
                    axis=1
                )
            
                # Create a date column (assuming the DataFrame index holds the dates).
                all_ts['date'] = pd.DatetimeIndex(pd.to_datetime(all_ts.index))
                y_columns = tuple(
                    f'{feed_id_to_string(feed_obj["feed_name"])}-count'
                    for feed_obj in filtered_feeds.values()
                )
                plost_chart = plost.area_chart(
                    data=all_ts,
                    stack='normalized',
                    x='date',
                    y=y_columns,
                    width=1200,
                    opacity=0.5,
                    height=200,
                    legend=None
                )             

            # Toggle Feed Digests
            if not session_state['render_feeds']:
                if st.button('Generate Digests for Selected Feeds', key='render-feeds-button'):
                    session_state['render_feeds'] = True
                    st.rerun()
            else:
                if st.button('Hide Feed Digests', key='render-feeds-button'):
                    session_state['render_feeds'] = False
                    st.rerun()

            # Show feed cards
            if session_state['render_feeds']:
                st.write(f'`Generating digests for {len(selected_feeds)} feeds...`')
                st.write('----------------------')

                if len(selected_feeds) <= 25:
                    for idx, feed_id in enumerate(selected_feeds):
                        render_feed_card(session_state["feeds"][tuple(feed_id)], idx, session_state)
                        st.write('---------')
                else:
                    st.error(
                        'Digests can be generated for max 25 selected feeds concurrently. '
                        'Please filter or deselect some items.'
                    )
        else:
            if len(session_state['signals']) == 0:
                renderer.write('### YOU HAVEN\'T ADDED ANY SIGNALS YET')


def main():
    hide_menu_and_footer()
    session_state = get_session_state()
    render_main_area(session_state)

if __name__ == '__main__':
    main()
