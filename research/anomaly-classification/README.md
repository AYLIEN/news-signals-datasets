# Anomaly Classification with News Signals Datasets


### Setup

Activate a new conda environment and run:
```shell
make dev
```

To run `make llm_experiments`, additionally install:
```
pip install -r llm-requirements.txt
```

Ensure you have these folders 

```
data/classification-datasets
data/models
data/predictions
```

Otherwise make them

```
mkdir -p data/classification-datasets
mkdir -p data/models
mkdir -p data/predictions
```

If you are running on a Mac system, Remember to change config files accordingly
From
```
"device": "cuda"
```
To
```
"device": "cpu"
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