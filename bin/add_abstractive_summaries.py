import argparse
import json
import logging
import os
import sys

from pathlib import Path

from news_signals.signals_dataset import SignalsDataset
from news_signals.dataset_transformations import get_dataset_transform
from news_signals.log import create_logger
from news_signals.summarization import Summary

from hybrid_mds import HybridSummarizer

here = Path(os.path.abspath(__file__))
model_path = here.parent.parent / 'research/hybrid-mds-wcep/models/ext-bart-large/checkpoint-20000'


logger = create_logger(__name__, level=logging.INFO)


class SummarizerWrapper:
    def __init__(self, summarizer):
        self.summarizer = summarizer 

    def __call__(self, articles, **kwargs):
        if len(articles) == 0:
            return Summary(summary=None)
        else:
            output = self.summarizer(articles)
            return Summary(summary=output['abstractive_summary'])


def main(args):
    logger.info('loading summarizer model')
    summarizer = SummarizerWrapper(HybridSummarizer(model_path))
    summarization_params = {}
    
    logger.info('loading dataset')
    dataset = SignalsDataset.load(args.input_dataset_path)
    
    
    def transform(signal):
        signal.summarize(
            summarizer=summarizer,
            summarization_params=summarization_params,
            cache_summaries=True,
            overwrite_existing=True
        )
        return signal

    logger.info('generating summaries')
    dataset.map(transform)
    
    
    # add_abstractive_summaries(dataset)
    # output_dataset_path = Path(args.output_dataset_path)
    # if str(output_dataset_path).endswith('.tar.gz'):
    #     dataset.save(output_dataset_path, overwrite=True, compress=True)
    # else:
    #     dataset.save(output_dataset_path, overwrite=True)
    



def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--input-dataset-path',
        required=True,
    )
    parser.add_argument(
        '--output-dataset-path',
    )
    return parser.parse_args()


if __name__ == '__main__':
    main(parse_args())
