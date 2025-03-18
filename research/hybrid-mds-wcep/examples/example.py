import json
from pathlib import Path
from hybrid_mds import HybridSummarizer
from hybrid_mds.extractive import CentroidExtractiveSummarizer


def main():
    
    articles = [json.loads(line) for line in Path('examples/articles.jsonl').open()]
    # ext = CentroidExtractiveSummarizer()
    # summary = ext(articles)
    # print(summary)
    model_path = 'models/ext-bart-large/checkpoint-20000'
    summarizer = HybridSummarizer(model_path)
    output = summarizer(articles)
    print('EXTRACTIVE:')
    for s in output['extracted_sentences']:
        print(s)
    print()
    print('ABSTRACTIVE:')
    print(output['abstractive_summary'])


if __name__ == '__main__':
    main()
