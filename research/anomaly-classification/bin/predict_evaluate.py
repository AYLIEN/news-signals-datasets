import argparse
import json
import pandas as pd
import logging
from pathlib import Path

from ac.data_utils import df_to_dataset
from ac.config import load_config
from ac.log import create_logger
from ac.utils import load_variable
from ac.evaluation import evaluate
from ac.transformer_classifier import TransformerClassifier, TransformerClassifierConfig
from ac.llm_classifier import LLMClassifier, LLMClassifierConfig
from ac.random_baseline import RandomClassifier, RandomClassifierConfig
from ac.sparse_classifiers import (
    SparseRandomForestClassifier,
    SparseRandomForestClassifierConfig,
    SparseLinearClassifier,
    SparseLinearClassifierConfig
)


MODEL_CLASSES = [
    TransformerClassifier,
    LLMClassifier,
    SparseLinearClassifier,
    SparseRandomForestClassifier,
    RandomClassifier,
]
CONFIG_CLASSES = [
    TransformerClassifierConfig,
    LLMClassifierConfig,
    SparseLinearClassifierConfig,
    SparseRandomForestClassifierConfig,
    RandomClassifierConfig
]


logger = create_logger(__name__,  level=logging.INFO)


def main(args, unknown_args):
    model_cls = load_variable(args.model_class, MODEL_CLASSES)
    config_cls = load_variable(args.model_class + 'Config', CONFIG_CLASSES)
    config = load_config(config_cls, args.config, unknown_args)

    logger.info(f'Test dataset: {Path(args.dataset_path).name}')
    logger.info("Configuration:")
    logger.info("\n" + str(config))
    
    classifier = model_cls.load(config, args.model_path)
    dataset_df = pd.read_parquet(args.dataset_path)
    dataset = df_to_dataset(dataset_df)[args.dataset_split]
    examples = [dataset[i] for i in range(dataset.num_rows)]
    if args.first_k is not None:
        examples = examples[:args.first_k]

    # Note this is aspect-based text classification, i.e.
    # each example is a {"text": "...", "aspect": "..."} item.    
    logger.info("Generating predictions")
    predictions = classifier.predict(examples)

    if args.save_with_dataset:
        for p, x in zip(predictions, examples):
            p.update(x)
    
    logger.info("Evaluating results")

    results = evaluate(predictions, examples)

    if args.output_path is not None:
        output_dir = Path(args.output_path)
        output_dir.mkdir(exist_ok=True)
        config.save(output_dir / "config.json")
    
        with open(output_dir / "predictions.jsonl", "w") as f:
            f.write("\n".join([json.dumps(x) for x in predictions]))

        with open(output_dir / "evaluation.json", "w") as f:
            json.dump(results, f)


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
        help='path to model config json file for inference-time settings'
    ),
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='path to dataset to run predictions on'
    ),
    parser.add_argument(
        '--model-path',
        required=False,
        help='path to directory where trained model and metadata are stored'
    ),
    parser.add_argument(
        '--output-path',
        required=False,
        help='output dir for storing predictions and config'
    ),    
    parser.add_argument(
        '--dataset-split',
        required=True,
        help='train, dev, or test'
    ),
    parser.add_argument(
        '--save-with-dataset',
        action='store_true',        
        help='whether to save predictions and dataset together'
    ),    
    parser.add_argument(
        '--first-k',
        required=False,
        default=None,
        type=int,
        help='if specified, only run prediction/eval on first k dataset items'
    ),    
    args, unknown_args = parser.parse_known_args()    
    return args, unknown_args


if __name__ == '__main__':
    main(*parse_args())

