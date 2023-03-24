# News Signals

### Example Colab Notebooks

These notebooks let you try out `news-signals` without installing anything!

- [Find Emerging Categories in a Newsfeed](https://drive.google.com/file/d/1NVBdCfKL3qdSIGITcLsewGETTqn-V9j6/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=oJa-xWusaCQ)       
- [Searching the Aylien NewsAPI Using An Entity Name](https://drive.google.com/file/d/1zKCSjWqxRJCPWBaGKXt5oQwkOzT8aDSg/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=HdoOiMXOrQ8)      
- [An Overview of News Signals Datasets](https://drive.google.com/file/d/1zM4J3jFA9v2LDTFKOpaa3EUUGhOdQieo/view?usp=share_link) - [Video](https://www.youtube.com/watch?v=wOMDSMxVUHY)

The Colab notebooks above are also available as jupyter notebooks in [research/news-signals-example-notebooks](research/news-signals-example-notebooks)


## Quickstart


#### Install news-signals in a new environment
```
conda create -n test-signals-pypi python=3.8
conda activate test-signals-pypi

pip install news-signals
```

#### Look at a sample dataset

Do `pip install jupyter` in your environment to run this code
in a jupyter notebook or in ipython, or just type `python` in your terminal. 
```
from news_signals.signals_dataset import SignalsDataset

# nasdaq100 sample dataset
dataset_url = 'https://drive.google.com/uc?id=150mfU2YA4ScfTlJvO6Duzto4aT_Q7K3D'

dataset = SignalsDataset.load(dataset_url)
```

Now try:
```
import matplotlib.pyplot as plt


fig = dataset.plot()
plt.show()
```

## Installation from source

#### Install news-signals in a new environment

Run `conda create -n news-signals python=3.8` if you're using Anaconda, alternatively `python3.8 -m venv news-signals` or similar.
Note python>=3.8 is required.

```
source activate news-signals
git clone https://github.com/AYLIEN/news-signals-datasets.git
cd news-signals-datasets
pip install -r requirements.txt
pip install -e . # install in editable mode
make test   # run tests
```

## Generating a new Dataset

```shell

python bin/generate_dataset.py \
    --start 2022/01/01 \
    --end 2022/02/01 \
    --input-csv resources/test/nasdaq100.small.csv \
    --id-field "Wikidata ID" \
    --name-field "Wikidata Label" \
    --output-dataset-dir sample_dataset_output

```

## Transforming a Dataset

```shell

python bin/transform_dataset.py \
    --input-dataset-dir sample_dataset_output \
    --config resources/default_transform_config.json

```
This adds anomaly scores, summary headlines and Wikimedia pageviews to each signal in a dataset (specified in [config file](resources/default_transform_config.json)). 
