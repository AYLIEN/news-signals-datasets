import argparse
import logging

from news_signals.signals_dataset import SignalsDataset
from news_signals.dataset_transformations import add_anomalies, add_wikimedia_pageviews

from ac.config import load_config, Config
from ac.log import create_logger
from ac.dataset_preprocessing import (
    remove_empty_samples,
    train_dev_test_splits,
    discretize_anomaly_scores,
    sample_balanced_labels,
    remove_leakage_across_splits,
    remove_empty_signals
)

logger = create_logger(__name__,  level=logging.INFO)


class ClassificationDatasetConfig(Config):
    def __init__(self, args, overwriting_args=None):
        super().__init__()
        self.register_param('train_ratio', float, 0.8)
        self.register_param('dev_ratio', float, 0.1)
        self.register_param('num_positive_train_samples', int, 10000)
        self.register_param('num_negative_train_samples', int, 10000)
        self.register_param('anomaly_threshold', float, 3.)
        self.register_param(
            'target_signal', str, 'volume',
            possible_values=['count', 'wikimedia_pageviews']
        )
        self.set_params_from_args(args, overwriting_args)


def main(args, unknown_args):
    config = load_config(ClassificationDatasetConfig, args.config, unknown_args)

    logger.info("Configuration:")
    logger.info(config)
    
    dataset = SignalsDataset.load(args.input_dataset_path)
    dataset = remove_empty_signals(dataset)

    if config.target_signal == 'wikimedia_pageviews':
        dataset = add_wikimedia_pageviews(dataset, overwrite_existing=True)
        for s in dataset.signals.values():
            s.timeseries_df['count'] = s.timeseries_df['wikimedia_pageviews']

    dataset = add_anomalies(dataset, overwrite_existing=True)
    df = dataset.df()

    # print('WIKI PAGEVIEWS:')
    # print(df['wikimedia_pageviews'].sum())
    # print('ANOMALIES')
    # print(df['anomalies'])

    logger.info("Removing samples with empty text")
    df = remove_empty_samples(df)

    logger.info("Splitting dataset into train/dev/test")
    df = train_dev_test_splits(
        df,
        shuffle=False,
        random_state=42,
        train_ratio=config.train_ratio,
        dev_ratio=config.dev_ratio,
    )

    # TODO: do this during previous step somehow to make label counts more precise
    logger.info("Remove shared items between train/dev/test")
    df = remove_leakage_across_splits(
        df,
        removal_sequence=[
            ("test", "train"),
            ("test", "dev"),
            ("dev", "train")
        ],
        row_to_key = lambda x: f"{x['summary']['summary']} {x['signal_name']}"
    )

    logger.info("Creating binary labels for anomaly detection")
    df = discretize_anomaly_scores(
        df,
        threshold=config.anomaly_threshold
    )
    logger.info("Sample balanced labels for train split")
    df = sample_balanced_labels(
        df,
        target_col="is_anomaly",
        label_counts={
            0: config.num_negative_train_samples,
            1: config.num_positive_train_samples
        },
        selected_split_groups=["train"]
    )

    logger.info("Created dataset with these splits:")
    for split in df["split_group"].unique():
        logger.info(f"{split}: {len(df[df['split_group']==split])}")
    logger.info("Saving to parquet file")

    # change lists from np.ndarray to list
    # to prevent parquet saving error

    def stories_np_to_list(row):
        return row['stories'].tolist()
    df['stories'] = df.apply(stories_np_to_list, axis=1)

    # hard-coding summary here as concatenated headlines
    def create_headline_summary(row):
        summary = row['summary'].copy()
        if row['stories'] is not None:
            headlines = [s['title'] for s in row['stories']]
            headlines = list(set(headlines))
            summary['summary'] = '\n'.join(headlines)
        return summary

    df['summary'] = df.apply(create_headline_summary, axis=1)
    df.to_parquet(path=args.output_dataset_path)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dataset-path',
        required=True,
    )
    parser.add_argument(
        '--output-dataset-path',
    )
    parser.add_argument(
        '--config',
        required=True,
    )
    args, unknown_args = parser.parse_known_args()
    return args, unknown_args


if __name__ == '__main__':
    main(*parse_args())
