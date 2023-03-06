# News Signals Documentation

Welcome to News Signals Datasets!
Think of signals as wrappers around dataframes that make it easy to work with timeseries
data combined with feeds of textual data. Explore these docs to find out how to use this library.

## Quickstart

```
conda create -n test-signals-pypi python=3.8

conda activate test-signals-pypi

pip install news-signals
```

```
from news_signals import signals

# Create a signal from a dataframe
df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02', '2020-01-03'], 'value': [1, 2, 3]})
signal = signals.DataFrameSignal(df, date_column='date', value_column='value')



```




# Welcome to MkDocs

For full documentation visit [mkdocs.org](https://www.mkdocs.org).

## Commands

* `mkdocs new [dir-name]` - Create a new project.
* `mkdocs serve` - Start the live-reloading docs server.
* `mkdocs build` - Build the documentation site.
* `mkdocs -h` - Print help message and exit.

## Project layout

    mkdocs.yml    # The configuration file.
    docs/
        index.md  # The documentation homepage.
        ...       # Other markdown pages, images and other files.
