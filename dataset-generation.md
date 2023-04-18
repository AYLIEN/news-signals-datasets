## Generating a new Signals Dataset

A signals dataset groups multiple signals together.
Signals dataset generation can be controlled via json config files. 

For example:

```shell
make create-dataset DATASET_CONFIG=resources/dataset-config-example.json
```

The config file specifies all relevant settings for a dataset, e.g. time range and csv file with list of entities. Here are the configurations of our example datasets:
- [dataset-config-nasdaq100.json](resources/dataset-config-nasdaq100.json)
- [dataset-config-smp500.json](resources/dataset-config-smp500.json)


The `make create-dataset` task runs [bin/generate_dataset.py](bin/generate_dataset.py) which has two steps:
1. Retrieve news articles and news volume signal for given entity for each day within specified time range, using the [generate_dataset](news_signals/signals_dataset.py#L409) function. This step takes most of the time because of the large number of NewsAPI calls with large response objects that need to be made. If this process breaks due to connection issues or similar, simply re-run the command and it picks up from where it was interrupted.
2. Apply a range of transformations on the dataset, e.g. adding summaries for each day or adding an anomaly time series (see [news_signals/dataset_transformations.py](news_signals/dataset_transformations.py)).


Example of a dataset generation config file:
```json
{
    "start": "2020-01-01",
    "end": "2023-01-01",
    "input": "resources/nasdaq100wiki.csv",
    "output_dataset_dir": "dataset-nasdaq100",
    "id_field": "Wikidata ID",
    "name_field": "Wikidata Label",
    "stories_per_day": 20,
    "compress": true, 
    "transformations": [
        {
            "transform": "add_anomalies",
            "params": {
                "overwrite_existing": true
            }
        },
        {
            "transform": "add_summaries",
            "params": {
                "summarizer": "CentralTitleSummarizer",
                "summarization_params": {},
                "overwrite_existing": true
            }
        },
        {
            "transform": "add_wikimedia_pageviews",
            "params": {
                "overwrite_existing": true
            }
        }   
    ]    
}
```

The above fields correspond to parameters of the [generate_dataset](news_signals/signals_dataset.py#L409) function. Additionally, the `transformations` specify dataset transformation functions available in [news_signals/dataset_transformations.py](news_signals/dataset_transformations.py).

The `transformations` field is optional. You can apply transformations in a post-processing step, using [bin/transform_dataset.py](bin/transform_dataset.py).

For now, the easiest way to add custom transformations is to implement these in [dataset_transformations.py](news_signals/dataset_transformations.py).

