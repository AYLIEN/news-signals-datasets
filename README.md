# News Signals

### Example Colab Notebooks

These notebooks let you try out `news-signals` without installing anything!
The Colab notebooks below are also available as jupyter notebooks in [research/news-signals-example-notebooks](research/news-signals-example-notebooks)

- [Create a Signals Dataset from a CSV of Entity Names](https://drive.google.com/file/d/1Czm6JxIVapE6tnu9mGzRK9j2NmC0blPR/view?usp=share_link)         
- [Find Emerging Categories in a Newsfeed](https://drive.google.com/file/d/1NVBdCfKL3qdSIGITcLsewGETTqn-V9j6/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=oJa-xWusaCQ)       
- [Searching the Aylien NewsAPI Using An Entity Name](https://drive.google.com/file/d/1zKCSjWqxRJCPWBaGKXt5oQwkOzT8aDSg/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=HdoOiMXOrQ8)      
- [An Overview of News Signals Datasets](https://drive.google.com/file/d/1zM4J3jFA9v2LDTFKOpaa3EUUGhOdQieo/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=wOMDSMxVUHY)


**2023-11-30: NEW**: [Create_Wikimedia_Signals.ipynb](research/news-signals-example-notebooks/Create_Wikimedia_Signals.ipynb) shows how to build and explore a `WikimediaSignal` which does require not a NewsAPI account and works out-of-the-box for anyone.

## Quickstart


#### Install news-signals in a new environment
```
conda create -n test-signals-pypi python=3.10
conda activate test-signals-pypi

pip install news-signals
```

#### Look at a sample dataset

Do `pip install jupyter` in your environment,

then run the code below 
in a jupyter notebook or in in the (i)python repl. 
```python
from news_signals.signals_dataset import SignalsDataset

# nasdaq100 sample dataset
dataset_url = 'https://drive.google.com/uc?id=150mfU2YA4ScfTlJvO6Duzto4aT_Q7K3D'

dataset = SignalsDataset.load(dataset_url)
```

Now try:
```python
import matplotlib.pyplot as plt


fig = dataset.plot()
plt.show()
```

See the [API Documentation](https://aylien.github.io/news-signals-datasets/) for more info.

## Installation from source

#### Install news-signals in a new environment

Run `conda create -n news-signals python=3.10` if you're using Anaconda, alternatively `python3.10 -m venv news-signals` or similar.
Note python>=3.8 is required.

```
source activate news-signals
git clone https://github.com/AYLIEN/news-signals-datasets.git
cd news-signals-datasets
pip install -r requirements.txt
pip install -e . # install in editable mode
make test   # run tests
```

## Setting up Aylien NewsAPI credentials

The news-signals library looks for environment variables called 
`'NEWSAPI_APP_ID'` and `'NEWSAPI_APP_KEY'` - these are used to authenticate to the NewsAPI. 

One way to set these variables up for local development is to 
Put your Aylien NewsAPI credentials in a file called `~/.aylienconfig`
`.aylienconfig`
```
app-id=<your-app-id>
app-key=<your-app-key>
```

Then put the following in your `.bashrc` or similar shell config file:
```
export NEWSAPI_APP_ID=$(cat ~/.aylienconfig | grep "app-id" | cut -d'=' -f2)
export NEWSAPI_APP_KEY=$(cat ~/.aylienconfig| grep "app-key" | cut -d'=' -f2)
```

## Generating a new Dataset

Generate a new signals dataset as follows:

```shell
make create-dataset DATASET_CONFIG=resources/dataset-config-example.json
```

The config file specifies all relevant settings for a dataset, e.g. time range. Some examples that we used to create our provided example datasets:
- [dataset-config-nasdaq100.json](resources/dataset-config-nasdaq100.json)
- [dataset-config-smp500.json](resources/dataset-config-smp500.json)


A more detailed guide on generating new datasets is here: [dataset-generation.md](dataset-generation.md)

## Anomaly Classification Experiments

These currently [live here](https://github.com/AYLIEN/news-signals-datasets/tree/add-anomaly-experiments/research/anomaly-classification).
