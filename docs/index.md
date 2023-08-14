# News Signals Documentation

Welcome to News Signals Datasets!
Think of signals as wrappers around dataframes that make it easy to work with timeseries
data combined with feeds of textual data. Explore these docs to find out how to use this library.

## Quickstart

```
# create an environment, for example:
conda create -n news-signals python=3.10
conda activate news-signals

# install news-signals
pip install news-signals
```

```
from news_signals import signals

# Create a signal from a dataframe
df = pd.DataFrame({'date': ['2020-01-01', '2020-01-02', '2020-01-03'], 'value': [1, 2, 3]})
signal = signals.DataFrameSignal(df, date_column='date', value_column='value')
```

## API Documentation

- [signals API][signals]
- [signals dataset API][signals_dataset]

[signals]: api/signals.md 
[signals_dataset]: api/signals_dataset.md


# MkDocs Info

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
