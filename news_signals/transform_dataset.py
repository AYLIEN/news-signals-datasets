import argparse
import json
import logging
from pathlib import Path

from news_signals.signals_dataset import SignalsDataset
from news_signals.dataset_transformations import get_dataset_transform
from news_signals.log import create_logger


logger = create_logger(__name__, level=logging.INFO)


def main(args):
    if args.output_dataset_path is None:
        output_dataset_path = Path(args.input_dataset_path)
    else:
        output_dataset_path = Path(args.output_dataset_path)

    with open(args.config) as f:
        config = json.load(f)

    if (args.input_dataset_path == args.output_dataset_path) or output_dataset_path.exists():
        confirm = input(
            f"Are you sure you want to modify the dataset in {output_dataset_path}? "
            "Alternatively, you can set output_dataset_path to a new directory. (y|n) "
        )
        if confirm != "y":
            logger.info("Aborting")
            return

    dataset = SignalsDataset.load(args.input_dataset_path)

    # config is a list of transformations
    for t in config:
        logger.info(f"Applying transformation to dataset: {t['transform']}")
        transform = get_dataset_transform(t['transform'])
        transform(dataset, **t['params'])

    if str(output_dataset_path).endswith('.tar.gz'):
        dataset.save(output_dataset_path, overwrite=True, compress=True)
    else:
        dataset.save(output_dataset_path, overwrite=True)


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
        help="JSON string with config"
    ),
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
