from transformers import AutoTokenizer
from transformers import BartForConditionalGeneration

from hybrid_mds.extractive import CentroidExtractiveSummarizer


class HybridSummarizer:
    def __init__(self, model_path) -> None:
        self.model = BartForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.extractive_summarizer = CentroidExtractiveSummarizer()

    def __call__(self, articles, text_field='body', title_field='title'):
        extracted_sents = self.extractive_summarizer(
            articles,
            max_len=5,
            len_type="sents",
            title_field=title_field,
            text_field=text_field
        )
        extractive_summary = " ".join(extracted_sents)
        input_ids = self.tokenizer(
            extractive_summary,
            return_tensors="pt",
            max_length=1024,
            truncation=True
        ).input_ids
        outputs = self.model.generate(input_ids, max_length=128)
        abstractive_summary = self.tokenizer.decode(
            outputs[0],
            skip_special_tokens=True
        )
        print('SUMMARY')
        print(abstractive_summary)
        return {
            'extracted_sentences': extracted_sents,
            'abstractive_summary': abstractive_summary 
        }
