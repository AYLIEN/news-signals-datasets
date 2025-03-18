import argparse

from news_signals.signals_dataset import SignalsDataset


def main(args):
    dataset = SignalsDataset.load(args.dataset_path)
    n_signals = 0
    n_articles = 0
    start_dates = []
    end_dates = []
    for signal in dataset.signals.values():
        start_dates.append(signal.start)
        end_dates.append(signal.end)
        for stories in signal.feeds_df['stories']:
            n_articles += len(stories)
        n_signals += 1
    
    start = min(start_dates)
    end = max(end_dates)

    print('start:', start)
    print('end:', end)
    print('num.signals:', n_signals)
    print('num. articles:', n_articles)
    

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-path',
        required=True,
        help='path to signals dataset'
    ),
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    main(parse_args())
