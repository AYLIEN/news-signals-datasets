import os
import json
import requests

import altair as alt
import pandas as pd
import streamlit as st

# Import modules from news_signals (ensure the package is installed)
from news_signals import newsapi, signals
from news_signals.anomaly_detection import BollingerAnomalyDetector
from news_signals.exogenous_signals import (
    WikidataRelatedEntitiesSearcher,
    entity_name_to_wikidata_id,
)

# Retrieve API keys from environment variables
NEWSAPI_APP_KEY = os.getenv("NEWSAPI_APP_KEY")
NEWSAPI_APP_ID = os.getenv("NEWSAPI_APP_ID")

if not NEWSAPI_APP_KEY or not NEWSAPI_APP_ID:
    st.error("API keys are not set. Please set the environment variables.")

# Initialize the newsapi headers
newsapi.set_headers(NEWSAPI_APP_ID, NEWSAPI_APP_KEY)


# -------------------------
# Helper Functions
# -------------------------

def sanitize_entity(entity):
    """Sanitize an entity name for use as a column name."""
    return entity.replace(":", "_").replace(" ", "_")


def fetch_news_timeseries(entity, start_date, end_date):
    """
    Fetch news volume time series data for a given entity using Aylien News API.
    """
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
    response = requests.get(
        "https://api.aylien.com/v6/news/time_series",
        params=params,
        headers=headers
    )
    if response.status_code == 200:
        data = response.json()
        time_series = data.get("time_series", [])
        if time_series:
            df = pd.DataFrame(time_series)
            df["published_at"] = pd.to_datetime(df["published_at"])
            df.set_index("published_at", inplace=True)
            safe_entity = sanitize_entity(entity)
            df.rename(columns={"count": f"news_volume_{safe_entity}"}, inplace=True)
            return df
        else:
            return None
    else:
        st.error(f"Error fetching news time series: {response.status_code} and {response.text}")
        return None


def fetch_stock_timeseries(entity_label, entity_id, stock, start_date, end_date):
    """
    Use news_signals to retrieve a financial timeseries for the given stock.
    The AylienSignal is instantiated using the provided entity (for example, as a feature).
    """
    signal = signals.AylienSignal(
        name=entity_label,
        params={"entity_ids": [entity_id]}
    )
    ts_signal = signal(str(start_date), str(end_date))
    ts_signal.add_yfinance_timeseries(
        ticker=stock,
        columns=["Open", "Close", "Volume", "High", "Low", "RSI"]
    )
    return ts_signal.timeseries_df


def plot_bollinger_anomalies_altair(df):
    """
    Compute Bollinger anomalies on the stock closing prices and plot them using Altair.
    The chart displays a line for closing prices with vertical dashed lines at anomaly dates.
    """
    col = "Close" if "Close" in df.columns else df.columns[0]
    filtered_df = df.dropna(subset=[col])

    bollinger_detector = BollingerAnomalyDetector(window=20, num_std=2.0)
    anomalies = bollinger_detector(history=filtered_df[col], series=filtered_df[col])
    anomalies = anomalies.reindex(filtered_df.index)

    df_line = filtered_df.reset_index()
    df_line = df_line.rename(columns={df_line.columns[0]: "Date"})
    df_line["Date"] = pd.to_datetime(df_line["Date"])

    anomaly_dates = filtered_df.index[anomalies > 0]
    df_anom = pd.DataFrame({"Date": anomaly_dates})
    df_anom["Date"] = pd.to_datetime(df_anom["Date"])

    line_chart = alt.Chart(df_line).mark_line(color="white").encode(
        x=alt.X("Date:T", title="Date"),
        y=alt.Y(f"{col}:Q", title="Price")
    ).properties(width=700, height=400)

    rule_chart = alt.Chart(df_anom).mark_rule(color="red", strokeDash=[4, 4]).encode(
        x=alt.X("Date:T", title="Date")
    )

    chart = line_chart + rule_chart
    st.altair_chart(chart, use_container_width=True)


def get_related_entities(entity_id):
    """
    Retrieve related Wikidata entities by performing a breadth-first search
    on the outgoing links of the entity's Wikidata page.
    Returns a dictionary mapping Wikidata IDs to human-readable labels.
    """
    related = WikidataRelatedEntitiesSearcher(entity_id, depth=1)
    return related


@st.cache_data(show_spinner=False)
def query_entity_instance(wikidata_id):
    """
    Query Wikidata to get the 'instance of' (P31) label for a given Wikidata ID.
    Returns the instance label if found; otherwise, returns "Other".
    """
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
        else:
            return "Other"
    else:
        return "Other"


def group_related_entities_by_id(related_entities):
    """
    Group related entities by their Wikidata instance type.

    For each entity in the input dictionary (mapping Wikidata ID -> label), this function
    queries Wikidata to obtain its "instance of" type (P31) and uses that type as the grouping key.
    If no type is found, the entity is grouped under "Other".
    
    Returns:
        A dictionary where keys are instance types (e.g., "human", "organization") and values are lists of labels.
    """
    grouped = {}
    for wikidata_id, label in related_entities.items():
        instance_type = query_entity_instance(wikidata_id)
        grouped.setdefault(instance_type, []).append(label)
    return grouped


def compute_entity_stock_correlation(news_df, stock_df):
    """
    Compute correlation between a news timeseries (resampled daily) and the stock's closing prices.
    Returns the correlation value.
    """
    news_daily = news_df.resample("D").sum()
    stock_daily = stock_df.resample("D").last()  # last available value of each day
    combined = pd.concat([news_daily, stock_daily], axis=1).dropna()

    news_col = combined.columns[0]
    if "Close" in combined.columns:
        stock_col = "Close"
    elif "close" in combined.columns:
        stock_col = "close"
    else:
        raise KeyError("No stock closing price column found in: " + str(combined.columns))

    corr = combined[news_col].corr(combined[stock_col])
    return corr


def analyze_related_entities_corr(related_entities, stock_df, start_date, end_date):
    """
    For each related entity, fetch its news volume timeseries,
    compute the correlation with the stock's closing price,
    and display the top 5 related entities by absolute correlation.
    In the displayed chart, both the news volume (in blue) and the financial
    time series (in red) are normalized and overlaid.
    """
    related_news_dict = {}
    if related_entities:
        for wikidata_id, rel_label in related_entities.items():
            df_rel = fetch_news_timeseries(rel_label, start_date, end_date)
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
                    
                    merged["news_norm"] = (merged[news_col] - merged[news_col].mean()) / merged[news_col].std()
                    merged["stock_norm"] = (merged[stock_col] - merged[stock_col].mean()) / merged[stock_col].std()
                    
                    melted = merged.melt(
                        id_vars=["Date"],
                        value_vars=["news_norm", "stock_norm"],
                        var_name="Series",
                        value_name="Normalized Value"
                    )
                    
                    color_scale = alt.Scale(domain=["news_norm", "stock_norm"], range=["blue", "red"])
                    
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


def hf_transformer_forecast(timeseries_df):
    """
    Placeholder function for integration with a Hugging Face time series transformer model.
    Replace with actual model inference as needed.
    """
    st.info("Placeholder")
    return None


# -------------------------
# Streamlit UI Layout
# -------------------------

st.title("Aylien News Signals Demo")
st.write(
    "This demo shows an integration of news volume time series, related entity retrieval via Wikidata, "
    "stock timeseries, and anomaly detection."
)

# Sidebar inputs for the demo parameters
st.sidebar.header("Input Parameters")
entity_input = st.sidebar.text_input("Entity Name", "Elon Musk")
stock_input = st.sidebar.text_input("Stock Ticker", "TSLA")
start_date = st.sidebar.text_input("Start Date (YYYY-MM-DD)", "2023-01-01")
end_date = st.sidebar.text_input("End Date (YYYY-MM-DD)", "2023-12-31")

if st.sidebar.button("Run Demo"):
    # --- Section 1: Main Entity News Volume ---
    st.header(f"News Volume Time Series: {entity_input}")
    news_df = fetch_news_timeseries(entity_input, start_date, end_date)
    if news_df is not None:
        st.line_chart(news_df)
    else:
        st.warning("No news data available for this entity.")

    # --- Section 2: Related Entities (Grouped by Instance) ---
    st.header("Related Entities")
    entity_id = entity_name_to_wikidata_id(entity_input)
    st.write(f"Using Wikidata ID for {entity_input}: {entity_id}")
    related_entities = get_related_entities(entity_id)
    if related_entities:
        grouped_entities = group_related_entities_by_id(related_entities)
        st.write("Related Entities (grouped by instance type):")
        for instance_type, names in grouped_entities.items():
            # Display as: <instance_type>: name1, name2, name3
            instance_type = instance_type.capitalize()
            st.write(f"**{instance_type}**: {', '.join(names)}")
    else:
        st.info("No related entities found.")

    # --- Section 3: Anomaly Detection ---
    st.header("Anomaly Detection")
    stock_df = fetch_stock_timeseries(entity_input, entity_id, stock_input, start_date, end_date)
    if stock_df is not None and not stock_df.empty:
        plot_bollinger_anomalies_altair(stock_df)
    else:
        st.error("No stock data available.")

    # --- Section 4: Main Entity Correlation Analysis ---
    st.header("Main Entity Correlation Analysis")
    if news_df is not None and stock_df is not None:
        news_df.index = pd.to_datetime(news_df.index)
        stock_df.index = pd.to_datetime(stock_df.index)
        try:
            corr_main = compute_entity_stock_correlation(news_df, stock_df)
            st.write(f"Correlation between {entity_input} news volume and stock closing price: **{corr_main:.2f}**")
        except Exception as e:
            st.write(f"Error computing main entity correlation: {e}")
    else:
        st.info("Insufficient data for main entity correlation analysis.")

    # --- Section 5: Related Entities Correlation Analysis ---
    st.header("Related Entities Correlation Analysis")
    analyze_related_entities_corr(related_entities, stock_df, start_date, end_date)

    # --- Section 6: HF Transformer Forecast (Placeholder) ---
    st.header("HF Transformer Forecast (Placeholder)")
    hf_transformer_forecast(stock_df)
