import os
import json
import requests
import logging
from datetime import datetime, timedelta
import altair as alt
import pandas as pd
import streamlit as st

from news_signals import newsapi, signals
from news_signals.anomaly_detection import BollingerAnomalyDetector
from news_signals.exogenous_signals import (
    WikidataRelatedEntitiesSearcher,
    entity_name_to_wikidata_id,
)
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential

st.set_page_config(
    page_title="Financial Demo",
    initial_sidebar_state="collapsed"
)

# Reduce logging for azure and urllib3 output
logging.getLogger("azure.core.pipeline.policies.http_logging_policy").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Hard-coded Azure credentials (replace these with your actual values)
azure_endpoint = "https://nlp-hub-1.openai.azure.com/openai/deployments/nlp-hub-1-dev-1-gpt4o"
azure_api_key = os.getenv("AZURE_OPENAI_API_KEY")  # Or set a hard-coded key for testing
deployment_name = "nlp-hub-1-dev-1-gpt4o"  # Your deployment name (model)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    force=True
)

NEWSAPI_APP_KEY = os.getenv("NEWSAPI_APP_KEY")
NEWSAPI_APP_ID = os.getenv("NEWSAPI_APP_ID")

if not NEWSAPI_APP_KEY or not NEWSAPI_APP_ID:
    st.error("API keys are not set. Please set the environment variables.")
else:
    newsapi.set_headers(NEWSAPI_APP_ID, NEWSAPI_APP_KEY)

# -------------------------------------------------------------------
# Deep Linking: Read default parameters from the URL using st.query_params
# st.query_params behaves like a dictionary.
default_entity = st.query_params.get("entity", "Jensen Huang")
default_stock = st.query_params.get("stock", "NVDA")
default_start = st.query_params.get("start", "2024-01-01")
default_end = st.query_params.get("end", "2025-01-01")
default_use_azure = st.query_params.get("use_azure", "False") == "True"

def get_session_state():
    """Initialize and return session state."""
    if 'run_demo' not in st.session_state:
        st.session_state['run_demo'] = False
    if 'news_df' not in st.session_state:
        st.session_state['news_df'] = None
    if 'stock_df' not in st.session_state:
        st.session_state['stock_df'] = None
    if 'entity_id' not in st.session_state:
        st.session_state['entity_id'] = None
    return st.session_state

def sanitize_entity(entity):
    """Sanitize an entity name for use as a column name."""
    return entity.replace(":", "_").replace(" ", "_")

def fetch_news_timeseries(entity, start_date, end_date):
    headers = {
        "X-AYLIEN-NewsAPI-Application-ID": NEWSAPI_APP_ID,
        "X-AYLIEN-NewsAPI-Application-Key": NEWSAPI_APP_KEY,
    }
    params = {
        "published_at.start": start_date + "T00:00:00.000Z",
        "published_at.end": end_date + "T23:59:59.999Z",
        "language": "(en)",
        "entities": "{{surface_forms:(" + json.dumps(entity) + ") AND overall_prominence:>=0.6}}",
    }
    response = requests.get("https://api.aylien.com/v6/news/time_series", params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        time_series = data.get("time_series", [])
        logging.debug("Received time series data: %s", time_series)
        if time_series:
            df = pd.DataFrame(time_series)
            df["published_at"] = pd.to_datetime(df["published_at"])
            df.set_index("published_at", inplace=True)
            safe_entity = sanitize_entity(entity)
            df.rename(columns={"count": f"news_volume_{safe_entity}"}, inplace=True)
            return df
        return None
    st.error(f"Error fetching news time series for entity '{entity}': {response.status_code} - {response.text}")
    return None

def fetch_corr_timeseries(entity, corr_entity, start_date, end_date):
    headers = {
        "X-AYLIEN-NewsAPI-Application-ID": NEWSAPI_APP_ID,
        "X-AYLIEN-NewsAPI-Application-Key": NEWSAPI_APP_KEY,
    }
    params = {
        "published_at.start": start_date + "T00:00:00.000Z",
        "published_at.end": end_date + "T23:59:59.999Z",
        "language": "(en)",
        "title": f'{json.dumps(entity)} AND {json.dumps(corr_entity)}'
    }
    response = requests.get("https://api.aylien.com/v6/news/time_series", params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        time_series = data.get("time_series", [])
        logging.debug("Received time series data: %s", time_series)
        if time_series:
            df = pd.DataFrame(time_series)
            df["published_at"] = pd.to_datetime(df["published_at"])
            df.set_index("published_at", inplace=True)
            safe_entity = sanitize_entity(entity)
            df.rename(columns={"count": f"news_volume_{safe_entity}"}, inplace=True)
            return df
        return None
    st.error(f"Error fetching news time series for entities '{entity}' and '{corr_entity}': {response.status_code} - {response.text}")
    return None

def fetch_stock_timeseries(entity_label, entity_id, stock, start_date, end_date):
    signal = signals.AylienSignal(name=entity_label, params={"entity_ids": [entity_id]})
    ts_signal = signal(str(start_date), str(end_date))
    ts_signal.add_yfinance_timeseries(ticker=stock, columns=["Open", "Close", "Volume", "High", "Low", "RSI"])
    return ts_signal.timeseries_df

def plot_bollinger_anomalies_altair(df):
    col = "Close" if "Close" in df.columns else df.columns[0]
    filtered_df = df.dropna(subset=[col])
    bollinger_detector = BollingerAnomalyDetector(window=40, num_std=3.0)
    anomalies = bollinger_detector(history=filtered_df[col], series=filtered_df[col])
    anomalies = anomalies.reindex(filtered_df.index)
    df_line = filtered_df.reset_index().rename(columns={filtered_df.reset_index().columns[0]: "Date"})
    df_line["Date"] = pd.to_datetime(df_line["Date"])
    anomaly_dates = filtered_df.index[anomalies > 0]
    df_anom = pd.DataFrame({"Date": anomaly_dates})
    df_anom["Date"] = pd.to_datetime(df_anom["Date"])
    line_chart = alt.Chart(df_line).mark_line(color="white").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(f"{col}:Q", title="Price")
    ).properties(width=700, height=400)
    rule_chart = alt.Chart(df_anom).mark_rule(color="red", strokeDash=[4, 4]).encode(x=alt.X("Date:T", title="Date"))
    chart = line_chart + rule_chart
    st.altair_chart(chart, use_container_width=True)

def get_related_entities(entity_id):
    related = WikidataRelatedEntitiesSearcher(entity_id, depth=1)
    return related

@st.cache_data(show_spinner=False)
def query_entity_instance(wikidata_id):
    query = f"""
    SELECT ?instanceOfLabel WHERE {{
      wd:{wikidata_id} wdt:P31 ?instanceOf .
      SERVICE wikibase:label {{ bd:serviceParam wikibase:language "en". }}
    }}
    LIMIT 1
    """
    url = "https://query.wikidata.org/sparql"
    headers = {"User-Agent": "StreamlitApp/1.0 (your_email@example.com)"}
    response = requests.get(url, params={"query": query, "format": "json"}, headers=headers)
    if response.status_code == 200:
        data = response.json()
        results = data.get("results", {}).get("bindings", [])
        if results:
            return results[0]["instanceOfLabel"]["value"]
        return "Other"
    return "Other"

def group_related_entities_by_id(related_entities):
    grouped = {}
    for wikidata_id, label in related_entities.items():
        instance_type = query_entity_instance(wikidata_id)
        grouped.setdefault(instance_type, []).append(label)
    return grouped

def compute_entity_stock_correlation(news_df, stock_df):
    news_daily = news_df.resample("D").sum()
    news_3day = news_daily.rolling(window=3, min_periods=3).sum()
    stock_daily = stock_df.resample("D").last()
    combined = pd.concat([news_3day, stock_daily], axis=1).dropna()
    news_col = combined.columns[0]
    if "Close" in combined.columns:
        stock_col = "Close"
    elif "close" in combined.columns:
        stock_col = "close"
    else:
        raise KeyError("No stock closing price column found in: " + str(combined.columns))
    corr = combined[news_col].corr(combined[stock_col])
    return corr

def analyze_related_entities_corr(entity_name, related_entities, stock_df, start_date, end_date):
    related_news_dict = {}
    if related_entities:
        for wikidata_id, rel_label in related_entities.items():
            df_rel = fetch_corr_timeseries(entity_name, rel_label, start_date, end_date)
            if df_rel is not None and not df_rel.empty:
                related_news_dict[rel_label] = df_rel
        if related_news_dict:
            correlations = {}
            for rel_label, df_rel in related_news_dict.items():
                df_rel.index = pd.to_datetime(df_rel.index)
                try:
                    corr_val = compute_entity_stock_correlation(df_rel, stock_df)
                    correlations[rel_label] = corr_val
                except Exception as e:
                    st.write(f"Error computing correlation for {rel_label}: {e}")
            correlations = {k: v for k, v in correlations.items() if pd.notnull(v)}
            if correlations:
                sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
                top5 = sorted_corr[:5]
                st.write("Top 5 related entities by absolute correlation:")
                for rel_label, corr_val in top5:
                    st.subheader(f"{rel_label}: Correlation = {corr_val:.2f}")
                    st.write("Normalized News Volume and Stock Closing Price:")
                    df_rel = related_news_dict[rel_label]
                    stock_col = "Close" if "Close" in stock_df.columns else "close"
                    news_col = df_rel.columns[0]
                    merged = pd.merge(df_rel, stock_df[[stock_col]], left_index=True, right_index=True, how="inner")
                    merged = merged.reset_index().rename(columns={"index": "Date"})
                    merged["Date"] = pd.to_datetime(merged["Date"])
                    merged["news_volume"] = (merged[news_col] - merged[news_col].mean()) / merged[news_col].std()
                    merged["stock"] = (merged[stock_col] - merged[stock_col].mean()) / merged[stock_col].std()
                    melted = merged.melt(
                        id_vars=["Date"],
                        value_vars=["news_volume", "stock"],
                        var_name="Series",
                        value_name="Normalized Value"
                    )
                    color_scale = alt.Scale(domain=["news_volume", "stock"], range=["blue", "red"])
                    chart = alt.Chart(melted).mark_line().encode(
                        x=alt.X("Date:T", title="Date"),
                        y=alt.Y("Normalized Value:Q", title="Normalized Value"),
                        color=alt.Color("Series:N", scale=color_scale, title="Series")
                    ).properties(width=700, height=400)
                    st.altair_chart(chart, use_container_width=True)
            else:
                st.info("No valid correlations computed for related entities.")
        else:
            st.info("No news data available for related entities.")
    else:
        st.info("No related entities found for correlation analysis.")

def get_anomaly_dates(df):
    col = "Close" if "Close" in df.columns else df.columns[0]
    filtered_df = df.dropna(subset=[col])
    bollinger_detector = BollingerAnomalyDetector(window=40, num_std=3.0)
    anomalies = bollinger_detector(history=filtered_df[col], series=filtered_df[col])
    anomalies = anomalies.reindex(filtered_df.index)
    anomaly_dates = filtered_df.index[anomalies > 0]
    return [date.strftime("%Y-%m-%d") for date in anomaly_dates]

def fetch_news_articles(entity, stock, start_date, end_date):
    headers = {
        "X-AYLIEN-NewsAPI-Application-ID": NEWSAPI_APP_ID,
        "X-AYLIEN-NewsAPI-Application-Key": NEWSAPI_APP_KEY,
    }
    params = {
        "published_at.start": start_date + "T00:00:00.000Z",
        "published_at.end": end_date + "T23:59:59.999Z",
        "language": "(en)",
        "entities": "{{surface_forms:(" + json.dumps(entity) + " OR " + json.dumps(stock) + ") AND overall_prominence:>=0.65}}",
        "per_page": 50
    }
    url = "https://api.aylien.com/v6/news/stories"
    response = requests.get(url, params=params, headers=headers)
    if response.status_code == 200:
        data = response.json()
        stories = data.get("stories", [])
        titles = [story.get("title", "") for story in stories if "title" in story]
        return titles
    st.error(f"Error fetching news articles. Status code: {response.status_code}, message: {response.text}")
    return []

def get_anomaly_explanation(anomaly_date, entity, news_titles):
    """
    Call the Azure OpenAI Chat endpoint to summarize the news titles and explain the anomaly
    for the given entity on the specified date.
    """
    prompt = (
        f"For the financial anomaly detected on {anomaly_date} for the stock {entity}, "
        "please summarize the following news titles from the past 3 days and explain possible reasons for the anomaly. "
        "Keep it short and concise:\n\n" + "\n".join(news_titles)
    )
    try:
        client = ChatCompletionsClient(
            endpoint=azure_endpoint,
            credential=AzureKeyCredential(azure_api_key),
        )
        system_message = SystemMessage(content="You are a helpful financial assistant.")
        user_message = UserMessage(content=prompt)
        response = client.complete(
            messages=[system_message, user_message],
            model=deployment_name,
            max_tokens=300,
            temperature=0.7,
            top_p=0.95
        )
        explanation = response.choices[0].message.content.strip()
        return explanation
    except Exception as e:
        logging.error(f"Error calling Azure OpenAI ChatCompletions: {e}")
        return None

@st.cache_data(show_spinner=False)
def cached_anomaly_explanation(anomaly_date, stock, news_titles):
    # Convert news_titles to a tuple to make it hashable for caching
    news_titles_tuple = tuple(news_titles)
    return get_anomaly_explanation(anomaly_date, stock, news_titles_tuple)

def hf_transformer_forecast(timeseries_df):
    st.info("Placeholder")
    return None

# ---------------------------
# Streamlit App Layout
# ---------------------------
st.title("News Signals Demo")
st.write(
    "This demo shows an integration of news volume time series, related entity retrieval via Wikidata, "
    "stock timeseries, anomaly detection, and anomaly explanation using Azure OpenAI."
)

# Sidebar: Input Parameters with defaults from query parameters (deep linking)
st.sidebar.header("Input Parameters")
entity_input = st.sidebar.text_input("Entity Name", default_entity, key="entity_input")
stock_input = st.sidebar.text_input("Stock Ticker", default_stock, key="stock_input")
start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", default_start, key="start_date")
end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", default_end, key="end_date")
use_azure = st.sidebar.checkbox("Generate Explanation using Azure OpenAI", value=default_use_azure, key="use_azure")

# When the Run Demo button is clicked, update the URL query parameters.
if st.sidebar.button("Run Demo"):
    # Update query parameters by assigning to st.query_params keys.
    st.query_params.entity = entity_input
    st.query_params.stock = stock_input
    st.query_params.start = start_date
    st.query_params.end = end_date
    st.query_params.use_azure = str(use_azure)
    
    state = get_session_state()
    state['run_demo'] = True
    state['news_df'] = fetch_news_timeseries(entity_input, start_date, end_date)
    entity_id = entity_name_to_wikidata_id(entity_input)
    state['entity_id'] = entity_id
    state['stock_df'] = fetch_stock_timeseries(entity_input, entity_id, stock_input, start_date, end_date)

# Use stored session state if demo has been run
state = get_session_state()
if state.get('run_demo', False):
    news_df = state.get('news_df')
    stock_df = state.get('stock_df')
    
    # Top Area: News Volume Timeseries
    st.header(f"News Volume Timeseries of {entity_input}")
    if news_df is not None:
        safe_entity = sanitize_entity(entity_input)
        news_col = f"news_volume_{safe_entity}"
        if news_col in news_df.columns:
            st.line_chart(news_df[news_col])
        else:
            st.line_chart(news_df)
    else:
        st.warning("No news data available for this entity.")

    # Financial Timeseries
    st.header(f"{stock_input} Financial Timeseries")
    if stock_df is not None and not stock_df.empty:
        if "Close" in stock_df.columns:
            st.line_chart(stock_df["Close"])
        elif "close" in stock_df.columns:
            st.line_chart(stock_df["close"])
        else:
            st.line_chart(stock_df)
    else:
        st.error("No stock data available.")

    # Create Horizontal Tabs for Navigation
    tab1, tab2 = st.tabs(["Relation Correlation", "Anomaly Detection"])

    with tab1:
        st.header("Related Entities and Correlation Analysis")
        entity_id = state.get('entity_id', None)
        st.write(f"Using Wikidata ID for {entity_input}: {entity_id}")
        related_entities = get_related_entities(entity_id)
        if related_entities:
            grouped_entities = group_related_entities_by_id(related_entities)
            st.write("Related Entities (grouped by instance type):")
            for instance_type, names in grouped_entities.items():
                st.write(f"**{instance_type.capitalize()}**: {', '.join(names)}")
        else:
            st.info("No related entities found.")

        st.subheader("Main Entity Correlation Analysis")
        if news_df is not None and stock_df is not None:
            try:
                corr_main = compute_entity_stock_correlation(news_df, stock_df)
                st.write(f"Correlation between **{entity_input}** news volume and **{stock_input}**: **{corr_main:.2f}**")
            except Exception as e:
                st.write(f"Error computing main entity correlation: {e}")
        else:
            st.info("Insufficient data for main entity correlation analysis.")

        st.subheader("Related Entities Correlation Analysis")
        analyze_related_entities_corr(entity_input, related_entities, stock_df, start_date, end_date)

    with tab2:
        st.header("Anomaly Detection and Explanation")
        if stock_df is not None and not stock_df.empty:
            plot_bollinger_anomalies_altair(stock_df)
            anomaly_dates = get_anomaly_dates(stock_df)
            if anomaly_dates:
                selected_date = st.selectbox("Detected Anomaly Dates", anomaly_dates)
                try:
                    selected_dt = datetime.strptime(selected_date, "%Y-%m-%d")
                except Exception as e:
                    st.error(f"Error parsing date: {e}")
                    selected_dt = None
                st.write(f"Selected Anomaly Date: {selected_date}")
                if selected_dt:
                    window_start_dt = selected_dt - timedelta(days=5)
                    window_end_dt = selected_dt + timedelta(days=5)
                    window_start = window_start_dt.strftime("%Y-%m-%d")
                    window_end = window_end_dt.strftime("%Y-%m-%d")
                    st.write(f"Fetching news articles from {window_start} to {window_end}...")
                    news_titles = fetch_news_articles(entity_input, stock_input, window_start, window_end)
                    if not news_titles:
                        st.info("No news articles found for the selected window.")
                    else:
                        # Use cached anomaly explanation if Azure is enabled
                        if use_azure:
                            if not azure_api_key:
                                st.error("Azure OpenAI API key not provided. Displaying news titles only.")
                                st.write("### News Titles")
                                for title in news_titles:
                                    st.write("-", title)
                            else:
                                with st.spinner("Generating explanation..."):
                                    explanation = cached_anomaly_explanation(selected_date, stock_input, news_titles)
                                st.write("### Anomaly Explanation")
                                st.write(explanation)
                        else:
                            st.write("### News Titles")
                            for title in news_titles:
                                st.write("-", title)
            else:
                st.info("No anomalies detected.")
        else:
            st.error("No stock data available.")
