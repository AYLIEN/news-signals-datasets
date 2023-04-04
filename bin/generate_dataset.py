import argparse
from pathlib import Path

import arrow

from news_signals.signals_dataset import generate_dataset


def main(args):
    generate_dataset(
        input=Path(args.input_csv),
        output_dataset_dir=Path(args.output_dataset_dir),
        start=arrow.get(args.start).datetime,
        end=arrow.get(args.end).datetime,
        id_field=args.id_field,
        name_field=args.name_field,
        overwrite=args.overwrite,
        delete_tmp_files=True,
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--start',
        required=True,
        help="start date, e.g. 2020-1-1"
    )
    parser.add_argument(
        '--end',
        required=True,
        help="end date, e.g. 2021-1-1"
    )
    parser.add_argument(
        '--input-csv',
        required=True,
        help="csv file with entities"
    )
    parser.add_argument(
        '--id-field',
        default="Wikidata ID",
        help="column in csv which indicates Wikidata id"
    )
    parser.add_argument(
        '--name-field',
        default="Wikidata Label",
        help="column in csv which indicates Wikidata Label"
    )
    parser.add_argument(
        '--output-dataset-dir',
        required=True,
        help="dir where dataset is stored"
    )
    parser.add_argument(
        '--overwrite',
        action="store_true",
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
