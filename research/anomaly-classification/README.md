# Anomaly Classification with News Signals Datasets


### Setup

Activate a new conda environment and run:
```shell
make dev
```

Exception: to run `make llm_experiments`, make another env and install:
```
pip install -r llm-requirements.txt
pip install -e .
```



### Download datasets

News signals datasets:

```shell
make pull_signals_datasets
```

Classification datasets:
```shell
make pull_classification_datasets
```

### Create classification datasets

```shell
make create_classification_datasets
```

### Run experiments

```shell
make sparse_experiments
make transformer_experiments
```

```shell
make llm_experiments
```