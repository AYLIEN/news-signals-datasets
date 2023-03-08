# News Signals

Check out this [colab notebook](https://drive.google.com/file/d/1iTjjeSt1S5WF0jJItH31DRe2C3IkZvz5/view?usp=sharing) to see some of the things you can do with the news-signals library.


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
