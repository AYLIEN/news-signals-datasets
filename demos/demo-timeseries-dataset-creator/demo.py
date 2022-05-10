import datetime

import pandas as pd
import streamlit as st
import plost
from matplotlib.dates import DateFormatter

from aylien_ts_datasets.newsapi import retrieve_timeseries
from aylien_ts_datasets.numerai_signals import generate_numerai_signals


page_config = st.set_page_config(
    page_title="Timeseries Dataset Creator"
)


def get_session_state():
    # Initialize session state
    if not st.session_state.get('INIT', False):
        st.session_state['button_click_count'] = 0



    st.session_state['INIT'] = True
    return st.session_state


def main():
    session_state = get_session_state()
    st.write("#### Timeseries data demo")
    current_entity = st.sidebar.text_input(
        'Entity',
        key='current_entity'
    )
    session_state['entity'] = current_entity
    today = datetime.date.today()
    last_week = today - datetime.timedelta(days=7)
    dates = st.sidebar.date_input(
        'Select Date Range', value=(last_week, today), key="dates"
    )
    # if dates[0] < dates[1]:
    #     st.success('Start date: `%s`\n\nEnd date:`%s`' % (dates[0], dates[1]))
    # else:
    #     st.error('Error: End date must fall after start date.')

    if len(dates) == 2:
        session_state['start_date'] = dates[0]
        session_state['end_date'] = dates[1]
    news_uploaded_df=None
    uploaded_file = st.sidebar.file_uploader("upload csv", type="csv", accept_multiple_files=False, key="upload_csv")
    if uploaded_file is not None:
        news_uploaded_df = pd.read_csv(uploaded_file, usecols=["date", "count","published_at"])
    # get NewsAPI data as json and timeseries volume
    if st.sidebar.button("Get Timeseries"):
        # ok let's get the news volume for a symbol and add it to the df
        query = {
            "title": session_state["entity"],
            "per_page": 100,
            "published_at.start": str(session_state["start_date"]) + "T00:00:00Z",
            "published_at.end": str(session_state["end_date"]) + "T00:00:00Z",
            "sort_by": "relevance",
        }
        data = retrieve_timeseries(query, n_pages=100)
        news_df = pd.DataFrame(data['time_series']) if news_uploaded_df is None else news_uploaded_df
        format = '%Y-%m-%d'
        news_df['date'] = pd.to_datetime(news_df['published_at'], format=format).dt.date
        news_df.set_index('date', inplace=True)
        yfinance_df = generate_numerai_signals(session_state["entity"], session_state["start_date"])
        main_df = pd.concat([news_df, yfinance_df], axis=1, join="inner")
        main_df.index.names = ['date']
        df = main_df.reset_index()
        df['mnth_yr'] = df['date'].apply(lambda x: str(x))
        print(df.head())
        plost.line_chart(
            df,
            x='mnth_yr',  # The name of the column to use for the x axis.
            y=('count', 'price', 'RSI'),
            use_container_width=True  # The name of the column to use for the data itself.
        )

        @st.cache
        def convert_df(df):
            return df.to_csv(columns=['mnth_yr', 'date', 'count', 'price', 'RSI','published_at']).encode('utf-8')

        csv = convert_df(df)

        st.sidebar.download_button(
            "Download as csv",
            data=csv,
            file_name=f"data-{session_state['start_date']}-{session_state['end_date']}.csv",
            mime="text/csv",
            key='download-csv'
        )


if __name__ == '__main__':
    main()
