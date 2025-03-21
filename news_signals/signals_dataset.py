import base64
import json
import logging
import os
import shutil
import tarfile
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Union, Optional
from copy import deepcopy

import appdirs
import arrow
import gdown
import pandas as pd
import tqdm

import news_signals.signals as signals
import news_signals.newsapi as newsapi
from news_signals.data import (aylien_ts_to_df, arrow_to_aylien_date,
                               load_from_gcs, save_to_gcs)
from news_signals.aql_builder import params_to_aql
from news_signals.log import create_logger


logger = create_logger(__name__, level=logging.INFO)

MAX_BODY_TOKENS = 1000
DEFAULT_METADATA = {
    'name': 'News Signals Dataset'
}


class SignalsDataset:
    DEFAULT_CACHE_DIR = Path(appdirs.user_cache_dir('news-signals/datasets'))

    def __init__(self, signals=None, metadata=None):
        if metadata is None:
            metadata = {
                # default dataset name
                'name': 'News Signals Dataset'
            }
        else:
            assert 'name' in metadata, 'Dataset metadata must specify a name.'
        self.metadata = metadata

        if signals is None:
            signals = {}
        if type(signals) is not dict:
            signals = {s.id: s for s in signals}
            assert len(set([s.ts_column for s in signals.values()])) == 1, \
                'All signals in a dataset must have the same `ts_column` attribute.'
        self.signals = signals

    def update(self):
        raise NotImplementedError

    @classmethod
    def load(cls, dataset_path, cache_dir=None):
        # handle downloading from urls
        if type(dataset_path) is str and (
            dataset_path.startswith('https://drive.google.com')
            or dataset_path.startswith('gs://')
        ):
            basename = base64.b64encode(dataset_path.encode()).decode()
            if cache_dir is None:
                cache_dir = cls.DEFAULT_CACHE_DIR
            else:
                cache_dir = Path(cache_dir)

            local_dataset_dir = cache_dir / basename
            if not local_dataset_dir.exists():
                if dataset_path.startswith('https://drive.google.com'):
                    # folder vs file download from gdrive
                    if 'folders' in dataset_path:
                        local_dataset_dir = Path(cache_dir) / basename
                        local_dataset_dir.mkdir(parents=True, exist_ok=True)
                        logger.info(f'Downloading dataset from {dataset_path} to {local_dataset_dir}.')
                        status = gdown.download_folder(
                            url=dataset_path,
                            output=str(local_dataset_dir),
                            remaining_ok=True
                        )
                        dataset_path = local_dataset_dir
                    else:
                        local_dataset_path = Path(str(local_dataset_dir) + '.tar.gz')
                        logger.info(f'Downloading dataset from {dataset_path} to {local_dataset_path}.')
                        status = gdown.download(url=dataset_path, output=str(local_dataset_path))
                        assert status is not None, 'Download as file failed.'
                        dataset_path = local_dataset_path
                elif dataset_path.startswith('gs://'):
                    assert dataset_path.endswith('.tar.gz'), \
                        'Datasets stored in GCS currently must be in .tar.gz format'
                    local_dataset_path = Path(str(local_dataset_dir) + '.tar.gz')
                    bucket_name, blob_name = dataset_path.replace("gs://", "").split("/", 1)
                    ds_cache_dir = Path(os.path.dirname(local_dataset_path))
                    ds_cache_dir.mkdir(parents=True, exist_ok=True)
                    load_from_gcs(
                        bucket_name=bucket_name,
                        blob_name=blob_name,
                        local_dataset_path=local_dataset_path
                    )
                    dataset_path = local_dataset_path
            else:
                logger.info(f'Using cached dataset at {local_dataset_dir}.')
                dataset_path = local_dataset_dir

        # handle decompressing tar.gz
        dataset_path = Path(dataset_path)
        if str(dataset_path).endswith('.tar.gz') or dataset_path.with_suffix('.tar.gz').exists():
            # add .tar.gz suffix if dataset_path doesn't already have it
            if not str(dataset_path).endswith('.tar.gz'):
                dataset_path = dataset_path.with_suffix('.tar.gz')

            # check if dataset_path exists without .tar.gz suffix
            expected_dataset_path = Path(str(dataset_path).replace('.tar.gz', ''))
            # already decompressed
            if os.path.exists(expected_dataset_path):
                logger.info(f'Found decompressed dataset at {expected_dataset_path}, '
                            'not decompressing again.')
            else:
                # extract tar.gz to the same directory as the tar.gz is in
                with tarfile.open(dataset_path, 'r:gz') as tar:
                    common_path = os.path.commonpath(tar.getnames())
                    expected_dataset_path = dataset_path.parent / common_path
                    print(f'Extracting dataset to {expected_dataset_path}')
                    if not expected_dataset_path.exists():
                        tar.extractall(path=dataset_path.parent)

            dataset_path = expected_dataset_path

        dataset_signals = signals.Signal.load(dataset_path)
        if (dataset_path / 'metadata.json').is_file():
            metadata = read_json(dataset_path / 'metadata.json')
        else:
            metadata = None
        return cls(
            signals=dataset_signals,
            metadata=metadata
        )

    def save(self, dataset_path, compress=True, overwrite=False, gcs_bucket_name=None):
        if gcs_bucket_name is not None:
            assert compress, 'Datasets uploaded to GCS must be compressed.'
        dataset_path = Path(dataset_path)
        if (overwrite and dataset_path.exists()) and not dataset_path.is_dir():
            dataset_path.unlink()
        dataset_path.mkdir(parents=True, exist_ok=overwrite)
        for signal in self.signals.values():
            signal.save(dataset_path)
        write_json(
            self.metadata,
            dataset_path / 'metadata.json'
        )
        if compress:
            shutil.make_archive(
                base_name=str(dataset_path),
                root_dir=dataset_path.parent,
                base_dir=dataset_path.name,
                format='gztar'
            )
            if dataset_path.exists():
                shutil.rmtree(dataset_path)
            logger.info(f'Saved compressed dataset to {dataset_path}.tar.gz')
            if gcs_bucket_name is not None:
                save_to_gcs(
                    bucket_name=gcs_bucket_name,
                    source_file_name=f'{dataset_path}.tar.gz',
                    destination_blob_name=f'{dataset_path.name}.tar.gz'
                )
            return f'{dataset_path}.tar.gz'
        else:
            logger.info(
                f'Saved {len(self.signals)} signals in dataset to {dataset_path}.'
            )
            return dataset_path

    def aggregate_signal(self, name=None):
        if name is None:
            name = self.metadata['name']
        return signals.AggregateSignal(
            name=name,
            components=list(self.signals.values())
        )

    def plot(self, savedir=None, **kwargs):
        plot = self.aggregate_signal().plot(**kwargs)
        if savedir is not None:
            savedir = Path(savedir)
            savedir.mkdir(parents=True, exist_ok=True)
            fig = plot.get_figure()
            plot_file = savedir / f'{self.metadata["name"]}.png'
            fig.savefig(plot_file)
            logger.info(f"Saved plot to {plot_file}.")
        return plot

    def df(self, axis=0):
        """
        Return a long form view of all the signals in the dataset.
        TODO: memoize when signals are the same between calls
        """
        return pd.concat(
            [s.df for s in self.signals.values()],
            axis=axis
        )

    def corr(self, **kwargs):
        """
        Compute pairwise correlation of signals in the dataset.
        """
        return self.aggregate_signal().corr(**kwargs)

    def __getattr__(self, name):
        """
        Try to delegate to pandas if the attribute is not found on SignalsDataset.
        """
        try:
            df = self.df(axis=0)
            return getattr(df, name)
        except AttributeError:
            raise AttributeError(
                f"type object 'SignalsDataset' has no attribute '{name}'"
            )

    def generate_report(self):
        """
        Generate a report containing summary statistics about the dataset.
        """
        pass

    def __len__(self):
        return len(self.signals)

    def __getitem__(self, key):
        return self.signals[key]

    def __iter__(self):
        return iter(self.signals)

    def __contains__(self, key):
        return key in self.signals

    def __repr__(self):
        return f"SignalsDataset({self.signals})"

    def __str__(self):
        return f"SignalsDataset({self.signals})"

    def items(self):
        return self.signals.items()

    def keys(self):
        return self.signals.keys()

    def values(self):
        return self.signals.values()

    def map(self, func):
        """
        Note this is embarassingly parallel, should
        be done multithreaded
        """
        logger.info(
            f'applying function to {len(self)} signals in dataset'
        )
        for k, v in tqdm.tqdm(self.signals.items(), total=len(self)):
            self.signals[k] = func(v)


def read_json(filepath):
    with open(filepath) as f:
        obj = json.load(f)
    return obj


def write_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f)


def read_jsonl(filepath):
    with open(filepath) as f:
        for line in f:
            yield json.loads(line)


def write_jsonl(items, filepath, mode="a"):
    content = "\n".join([json.dumps(x) for x in items]) + "\n"
    with open(filepath, mode) as f:
        f.write(content)


def ask_rmdir(dirpath, msg, yes="y"):
    if dirpath.exists():
        if input(msg) == yes:
            shutil.rmtree(dirpath)


def make_aylien_newsapi_query(params, start, end, period="+1DAY"):
    _start = arrow_to_aylien_date(arrow.get(start))
    _end = arrow_to_aylien_date(arrow.get(end))
    aql = params_to_aql(params)
    new_params = deepcopy(params)
    new_params.update({
        "published_at.start": _start,
        "published_at.end": _end,
        "period": period,
        "language": "en",
        "aql": aql,
    })
    return new_params


def reduce_aylien_story(
        s,
        max_body_tokens=MAX_BODY_TOKENS,
        additional_fields=None
):
    if additional_fields is None:
        additional_fields = []
    body = " ".join(s["body"].split()[:max_body_tokens])
    smart_cats = extract_aylien_smart_tagger_categories(s)
    reduced = dict(
        {
            "title": s["title"],
            "body": body,
            "id": s["id"],
            "published_at": s["published_at"],
            "language": s["language"],
            "url": s["links"]["permalink"],
            "categories": s["categories"],
            "industries": s["industries"],
            "smart_tagger_categories": smart_cats,
            "media": s["media"],
            "clusters": s["clusters"]
        }, **{f: s[f] for f in additional_fields}
    )
    return reduced


def extract_aylien_smart_tagger_categories(s):
    category_items = []
    for c in s["categories"]:
        if c["taxonomy"] == "aylien":
            item = {
                "score": c["score"],
                "id": c["id"]
            }
            category_items.append(item)
    return category_items


def read_last_timestamp(filepath):
    """
    Identifies last bucket's timestamp from buckets_*.jsonl file.
    """
    if filepath.exists():
        timestamps = [
            arrow.get(b["timestamp"]).datetime
            for b in read_jsonl(filepath)
        ]
        last = max(timestamps, key=arrow.get)
        return last
    else:
        return None


def retrieve_and_write_stories(
    params_template: Dict,
    start: datetime,
    end: datetime,
    ts: List,
    output_path: Path,
    num_stories: int = 20,
    stories_endpoint=newsapi.retrieve_stories,
    post_process_story=None,
):
    time_to_volume = dict(
        (arrow.get(x["published_at"]).datetime, x["count"]) for x in ts
    )

    params_template['per_page'] = num_stories
    date_range = signals.Signal.date_range(start, end)
    start_end_tups = [
        (s, e) for s, e in zip(list(date_range), list(date_range)[1:])
    ]
    last_time = read_last_timestamp(output_path)
    passed_last = False

    for start, end in tqdm.tqdm(start_end_tups):

        if start == last_time:
            passed_last = True
        if last_time is not None and start <= last_time:
            continue
        # just sanity-checking that we observed last date in loop
        assert last_time is None or passed_last

        vol = time_to_volume[start]
        if vol > 0:
            params = make_aylien_newsapi_query(params_template, start, end)
            stories = stories_endpoint(params)
            if post_process_story is not None:
                stories = [post_process_story(s) for s in stories]
        else:
            stories = []
        output_item = {
            "timestamp": str(start),
            "stories": stories,
            "volume": vol
        }
        write_jsonl([output_item], output_path, "a")


def retrieve_and_write_timeseries(
    params,
    start,
    end,
    output_path,
    ts_endpoint=newsapi.retrieve_timeseries
) -> List:
    if not output_path.exists():
        params = make_aylien_newsapi_query(params, start, end)
        ts = ts_endpoint(params)
        write_json(ts, output_path)
    else:
        ts = read_json(output_path)
    return ts


def df_from_jsonl_buckets(path):
    story_bucket_records = []
    for b in read_jsonl(path):
        item = {"timestamp": b["timestamp"], "stories": b["stories"]}
        story_bucket_records.append(item)
    df = pd.DataFrame.from_records(
        story_bucket_records,
        index='timestamp'
    )
    return df


def signal_exists(signal, dataset_output_dir):
    return any(
        [f.name.startswith(signal.id) for f in dataset_output_dir.iterdir()]
    )


def generate_dataset(
    input: Union[List[signals.Signal], Path],
    output_dataset_dir: Path,
    start: datetime,
    end: datetime,
    gcs_bucket: Optional[str] = None,
    name_field: Optional[str] = None,
    id_field: Optional[str] = None,
    surface_form_field: Optional[str] = None,
    stories_per_day: int = 20,
    overwrite: bool = False,
    delete_tmp_files: bool = False,
    stories_endpoint=newsapi.retrieve_stories,
    ts_endpoint=newsapi.retrieve_timeseries,
    post_process_story=None,
    compress=True,
):
    """
    Turn a list of signals into a dataset by populating each signal with time
    series and stories using Aylien Newsapi endpoints.
    """
    if isinstance(input, Path):
        # this CSV should have a Wikidata ID and/or entity surface form and name for each entity
        assert id_field is not None or surface_form_field is not None, 'dataset generation from CSV requires an ID and/or surface form field'
        df = pd.read_csv(input)
        signals_ = []
        for x in df.to_dict(orient="records"):
            if name_field is None:
                assert id_field is not None, 'if name_field is None, id_field must be specified'
                name = x[id_field]
            else:
                name = x[name_field]
            entity_ids = []
            surface_forms = []
            if id_field is not None:
                entity_ids.append(x[id_field])
            if surface_form_field is not None:
                surface_forms.append(x[surface_form_field])
            signal = signals.AylienSignal(
                name=name,
                params={
                    "entity_ids": entity_ids,
                    "entity_surface_forms": surface_forms
                }
            )
            signals_.append(signal)
    else:
        signals_ = input

    if overwrite and output_dataset_dir.exists():
        ask_rmdir(
            output_dataset_dir,
            msg=f"Are you sure you want to delete {output_dataset_dir} and "
            "start building dataset from scratch (y|n)? ",
        )
    output_dataset_dir.mkdir(parents=True, exist_ok=True)

    # optional, e.g. for reducing story fields
    if post_process_story is not None and isinstance(post_process_story, str):
        try:
            post_process_story = globals()[post_process_story]
        except KeyError:
            raise NotImplementedError(
                f"Unknown function for processing stories: {post_process_story}"
            )

    # Note this function creates queries from signals, but it
    # does not use the __call__ method implemented on Signal objects.
    for signal in tqdm.tqdm(signals_):
        if signal_exists(signal, output_dataset_dir):
            logger.info("signal exists already, skipping to next")
            continue

        stories_path = (
            output_dataset_dir / f"buckets_{signal.id}.jsonl"
        )
        ts_path = output_dataset_dir / f"timeseries_{signal.id}.jsonl"

        # TODO: pick a surface form vs. ID, or both
        params = signal.params

        # we save TS and stories to make continuation of the
        # dataset generation process easier if it gets interrupted
        # by an error.
        logger.info("retrieving time series")
        ts = retrieve_and_write_timeseries(
            params, start, end, ts_path,
            ts_endpoint=ts_endpoint
        )
        logger.info("retrieving stories")
        retrieve_and_write_stories(
            params,
            start, end,
            ts,
            stories_path,
            num_stories=stories_per_day,
            stories_endpoint=stories_endpoint,
            post_process_story=post_process_story
        )

        # now this signal is completely realized
        stories_df = df_from_jsonl_buckets(stories_path)
        ts_df = aylien_ts_to_df({"time_series": ts}, dt_index=True)
        signal.timeseries_df = ts_df
        signal.feeds_df = stories_df
        logger.info(f"saving signal: {signal.name}")
        signal.save(output_dataset_dir)
        # clear memory
        del signal.feeds_df, signal.timeseries_df

        # delete temporary files
        if delete_tmp_files:
            ts_path.unlink()
            stories_path.unlink()

    dataset = SignalsDataset.load(output_dataset_dir)
    if compress:
        shutil.rmtree(output_dataset_dir)
        dataset.save(
            output_dataset_dir,
            compress=compress,
            gcs_bucket_name=gcs_bucket
        )
    return dataset
