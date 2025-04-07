# Financial Demo

This demo showcases the integration of news volume time series, related entity retrieval via Wikidata, stock timeseries, anomaly detection, and anomaly explanation using Azure OpenAI.

## Setup

### Put in your NewsAPI key and ID 

```bash
export NEWSAPI_APP_KEY="<YOUR NEWSAPI KEY HERE>"
export NEWSAPI_APP_ID="<YOUR NEWSAPI ID HERE>"
```

### Create a New Conda Environment

To create a new conda environment for this demo, run the following command:

```bash
conda create --name financial-demo python=3.10
conda activate financial-demo
```

### Install Dependencies

Install the required Python packages using the `make dev` command:

```bash
make dev
```

This will install the packages listed in `requirements.txt`, including:

- `streamlit>=1.43.2`
- `pandas>=2.2.3`
- `altair>=5.5.0`
- `requests>=2.32.3`
- `azure-ai-inference>=1.0.0b9`
- `azure-core>=1.32.0`
- `news-signals>=0.8.2`
- `diskcache>=5.6.1`
- `yfinance>=0.2.54`

## Running the Demo

To run the demo, use the `make run` command:

```bash
make run
```

This will start the Streamlit application on port 9001 with the base URL path set to `finance-demo`.

## Features

### Deeplinking URL

The application supports deeplinking by reading default parameters from the URL using `st.query_params`. The parameters include:

- `entity`: Default is "Jensen Huang"
- `stock`: Default is "NVDA"
- `start`: Default is "2024-01-01"
- `end`: Default is "2025-01-01"
- `use_azure`: Default is "False"

You can copy the url and paste it in your browser (with the demo running) and get the same query parameters filled in

### Persistent Cache

The demo uses `diskcache` for persistent caching, storing results on disk in the `cache_dir`.

If an query has run before, all the saved data will be loaded back in. It stores all the timeseries and correlation results after running once and dynamically stores the results of the anomaly summaries and news titles based on if they have been generated before.

### LLM Usage

To use the `generate anomaly function`, please put in your OpenAI Azure endpoint key like so:

```bash
export AZURE_OPENAI_API_KEY="<YOUR OPENAI API KEY HERE>"
export AZURE_OPENAI_ENDPOINT="<YOUR OPENAI ENDPOINT HERE>"
export AZURE_OPENAI_DEPLOYMENT_NAME="<YOUR OPENAI DEPLOYMENT MODEL NAME HERE>"
```
The demo can be used without the AI features and will default to showing News Titles for anomalies unless provided OpenAI endpoint variables.

The demo is made to work with the gpt-4o model endpoints.

### Queries

The demo performs several queries, including:

- Fetching news time series and correlation time series from the News API.
- Fetching news titles from the News API for the anomaly explanations.
- Querying Wikidata using the News Signals library for the entity correlations searches.
- Using yfinance features using the News Signals library to get the financial data timeseries.
- Querying the Azure OpenAI endpoint if enabled to get anomaly explanations.

The NewsAPI query that fetches the news timeseries queries the `time_series` endpoint and the query passed is

```python
params = {
    "published_at.start": start_date + "T00:00:00.000Z",
    "published_at.end": end_date + "T23:59:59.999Z",
    "language": "(en)",
    "entities": "{{surface_forms:(" + json.dumps(entity) + ") AND overall_prominence:>=0.6}}",
}
```

The NewsAPI query that gets the correlation timeseries, the query passed to get the news volume timeseries of the related entities from the `time_series` endpoint is

```python
params = {
    "published_at.start": start_date + "T00:00:00.000Z",
    "published_at.end": end_date + "T23:59:59.999Z",
    "language": "(en)",
    "title": f'{json.dumps(entity)} AND {json.dumps(corr_entity)}'
}
```

The NewsAPI query that gets the news titles to display the anomaly detected dates, gets the new titles from a window of ± 5 days from the anomaly date using the `stories` endpoint.

```python
params = {
    "published_at.start": start_date + "T00:00:00.000Z",
    "published_at.end": end_date + "T23:59:59.999Z",
    "language": "(en)",
    "entities": "{{surface_forms:(" + json.dumps(entity) + " OR " + json.dumps(stock) + ") AND overall_prominence:>=0.65}}",
    "per_page": 50
}
```