import argparse
import json
import logging
from pathlib import Path

import arrow

from news_signals.signals_dataset import generate_dataset, reduce_aylien_story
from news_signals.dataset_transformations import get_dataset_transform
from news_signals.log import create_logger


logger = create_logger(__name__, level=logging.INFO)


def main(args):
    with open(args.config) as f:
        config = json.load(f)

    output_dataset_path = Path(config["output_dataset_dir"])
    logger.info(f"Beginning dataset generation with config {config}") 
    dataset = generate_dataset(
        input=Path(config["input"]),
        output_dataset_dir=output_dataset_path,
        start=arrow.get(config["start"]).datetime,
        end=arrow.get(config["end"]).datetime,
        stories_per_day=config["stories_per_day"],
        id_field=config["id_field"],
        name_field=config["name_field"],
        overwrite=args.overwrite,
        delete_tmp_files=True,
        compress=True,
        post_process_story=reduce_aylien_story
    )    

    if config.get("transformations"):
        for t in config["transformations"]:
            logger.info(f"Applying transformation to dataset: {t['transform']}")
            transform = get_dataset_transform(t['transform'])
            transform(dataset, **t['params'])
    
    dataset.save(output_dataset_path, overwrite=True, compress=True)
    logger.info(f"Finished dataset generation, dataset saved here: {output_dataset_path}.tar.gz")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        required=True,
        help="config json file containing all settings to create new dataset"
    ),
    parser.add_argument(
        '--overwrite',
        action="store_true",
        help="whether to overwrite previous dataset if present at output_dataset_dir"
    )    
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
