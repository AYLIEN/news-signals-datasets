import logging
import numpy as np

from ac.classifier import Classifier
from ac.evaluation import get_label_ratio
from ac.config import Config
from ac.log import create_logger


logger = create_logger(__name__,  level=logging.INFO)


class RandomClassifierConfig(Config):
    def __init__(self, args, overwriting_args=None):
        super().__init__()
        self.register_param('positive_prob_mode', str, 'target', possible_values=["target", "custom", "uniform"])
        self.register_param('custom_positive_prob', float, 0.1)
        self.set_params_from_args(args, overwriting_args)


class RandomClassifier(Classifier):
    def __init__(self, positive_prob_mode, custom_positive_prob):
        self.positive_prob_mode = positive_prob_mode
        self.custom_positive_prob = custom_positive_prob

    @staticmethod
    def load(config, model_path):
        return RandomClassifier(
            config.positive_prob_mode,
            config.custom_positive_prob
        )

    def predict(self, examples):
        if self.positive_prob_mode == "target":
            true_labels = [x["label"] for x in examples]
            true_positive_ratio = get_label_ratio(true_labels)
            predicted_labels = np.random.binomial(
                1, true_positive_ratio, size=len(examples)
            )
        elif self.positive_prob_mode == "uniform":
            predicted_labels = np.random.binomial(
                1, 0.5, size=len(examples)
            )
        else:
            assert self.positive_prob_mode == "custom"
            predicted_labels = np.random.binomial(
                1, self.custom_positive_prob, size=len(examples)
            )
        
        predictions = [{"predicted_label": int(l)} for l in predicted_labels]
        return predictions

    @classmethod
    def train(
        cls,
        config,
        dataset_df,
        output_path,
    ):
        raise NotImplementedError
