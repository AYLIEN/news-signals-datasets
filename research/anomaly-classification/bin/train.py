import logging
import argparse
import pandas as pd
from pathlib import Path

from ac.log import create_logger
from ac.utils import load_variable
from ac.config import load_config
from ac.transformer_classifier import TransformerClassifier, TransformerClassifierConfig
from ac.sparse_classifiers import (
    SparseRandomForestClassifier,
    SparseRandomForestClassifierConfig,
    SparseLinearClassifier,
    SparseLinearClassifierConfig
)

MODEL_CLASSES = [
    TransformerClassifier,
    SparseLinearClassifier,
    SparseRandomForestClassifier
]
CONFIG_CLASSES = [
    TransformerClassifierConfig,
    SparseLinearClassifierConfig,
    SparseRandomForestClassifierConfig
]

logger = create_logger(__name__,  level=logging.INFO)


def main(args, unknown_args):

    model_cls = load_variable(args.model_class, MODEL_CLASSES)
    config_cls = load_variable(args.model_class + 'Config', CONFIG_CLASSES)
    config = load_config(config_cls, args.config, unknown_args)
    logger.info(f'Training dataset: {Path(args.dataset_path).name}')
    logger.info("Configuration:")
    logger.info(config)
    dataset_df = pd.read_parquet(args.dataset_path)
    model_cls.train(
        config=config,
        dataset_df=dataset_df,
        output_path=args.output_path
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--model-class',
        required=True,
        help='class of the trained model'
    ),
    parser.add_argument(
        '--config',
        required=True,
        help='path to config json file with model settings'
    ),
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='path to .parquet file containing train/dev/test data'
    ),
    parser.add_argument(
        '--output-path',
        required=True,
        help='path to directory for storing trained model and metadata'
    ),        
    args, unknown_args = parser.parse_known_args()    
    return args, unknown_args


if __name__ == '__main__':
    main(*parse_args())
