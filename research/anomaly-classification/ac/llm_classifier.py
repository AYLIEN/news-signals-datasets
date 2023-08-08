import logging

from jinja2 import Template

from ac.classifier import Classifier
from ac.config import Config
from ac.log import create_logger


try:
    from vllm import LLM, SamplingParams
except ImportError:
    pass


logger = create_logger(__name__,  level=logging.INFO)



class LLMClassifierConfig(Config):
    def __init__(self, args, overwriting_args=None):
        super().__init__()
        self.register_param('model_id', str, 'meta-llama/Llama-2-13b')
        self.register_param('prompt_template', str)
        self.register_param('temperature', float, 0.1)
        self.register_param('top_p', float, 0.95)
        self.register_param('max_tokens', int, 100)
        self.set_params_from_args(args, overwriting_args)


class LLMClassifier(Classifier):
    def __init__(self, llm, prompt_template, sampling_params):
        self.llm = llm
        self.prompt_template = prompt_template
        self.sampling_params = sampling_params

    @staticmethod
    def load(config, model_path):
        sampling_params = SamplingParams(
            temperature=config.temperature,
            top_p=config.top_p,
            max_tokens=config.top_p
        )
        llm = LLM(config.model_id)
        return LLMClassifier(
            llm,
            config.prompt_template,
            sampling_params,
        )
    
    def predict(self, examples):
        predictions = []
        for x in examples:
            prompt = Template(self.prompt_template).render(
                aspect=x['aspect'],
                text=x['text']
            )
            output = self.llm.generate([prompt], self.sampling_params)[0]
            generated_text = output.outputs[0].text.lower()
            if generated_text.startswith('yes'):
                predicted_label = 1
            else:
                predicted_label = 0
            pred_item = {
                'predicted_label': predicted_label,
            }
            predictions.append(pred_item)
        return predictions

    @classmethod
    def train(
        cls,
        config,
        dataset_df,
        output_path,
    ):
        raise NotImplementedError