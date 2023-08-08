import logging
import functools
import glob
from pathlib import Path
from pprint import pprint

import tqdm
import torch
from torch import nn
import evaluate
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    pipeline
)

from ac.classifier import Classifier
from ac.data_utils import df_to_dataset
from ac.log import create_logger
from ac.config import Config


logger = create_logger(__name__,  level=logging.INFO)



def create_input_text(text, aspect):
    return f"{aspect}</s>{text}"


def _preprocess_func(examples, tokenizer):
    input_texts = []
    for i in range(len(examples["text"])):
        text = examples["text"][i]
        aspect = examples["aspect"][i]
        input_text = create_input_text(text, aspect)        
        input_texts.append(input_text)
    return tokenizer(input_texts, truncation=True, padding=True, max_length=512)


def _compute_metrics(eval_pred, metric):
    predictions, labels = eval_pred
    predictions = predictions.argmax(1)
    return metric.compute(predictions=predictions, references=labels)


class SkipConnectionClassificationHead(nn.Module):
    """
    Head for sentence-level classification tasks, modified by adding skip-connection
    to prevent information loss when training a randomly initialised layer with
    embeddings of a pre-trained model.
    """

    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x_input = features[:, 0, :]  # take <s> token (equiv. to [CLS])
        x = self.dropout(x_input)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        # only modification of original head:
        x = self.out_proj(x + x_input)
        return x


class TransformerClassifierConfig(Config):

    def __init__(self, args, overwriting_args=None):
        super().__init__()
        self.register_param('model_id', str, 'distilroberta-base')
        self.register_param('epochs', int, 3)
        self.register_param('batch_size', int, 3)
        self.register_param('use_skip_connection', bool, False)
        self.register_param('metric_name', str, 'f1')
        self.register_param('device', str, 'cuda', possible_values=('cuda', 'cpu', 'mps:0'))
        self.set_params_from_args(args, overwriting_args)


class TransformerClassifier(Classifier):
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @staticmethod
    def load(config, model_path):
        """
        Loads classifier from model directory.
        """
        # TODO: need checkpoint selection logic if we have more than one
        model_dir = Path(model_path)
        model_path = glob.glob(f"{model_dir}/model/checkpoint*")[0]
        tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, local_files_only=True
        ).to(config.device)
        return TransformerClassifier(model, tokenizer)

    def predict(self, examples, batch_size=4, **kwargs):
        model_pipeline = pipeline(
            "text-classification",
            model=self.model,
            tokenizer=self.tokenizer,
            device=0,
            batch_size=batch_size
        )
        tokenizer_kwargs = {
            'padding': True,
            'truncation': True,
            'max_length': 512
        }
        input_texts = [
            create_input_text(text=x["text"], aspect=x["aspect"])
            for x in examples
        ]
        # predictions = model_pipeline(input_texts, **tokenizer_kwargs)
        label_map = {
            "LABEL_0": 0,
            "LABEL_1": 1
        }
        predictions = []
        for out in tqdm.tqdm(model_pipeline(input_texts, **tokenizer_kwargs)):
            pred_item = {
                "predicted_label": label_map[out["label"]],
                "score": out["score"]
            }
            predictions.append(pred_item)
        # predictions = [
        #     {
        #         "predicted_label": label_map[x["label"]],
        #         "score": x["score"]
        #     }
        #     for x in predictions
        # ]
        return predictions
    
    @classmethod
    def train(
        cls,
        config,
        dataset_df,
        output_path,  
    ):
        model_dir = Path(output_path)
        model_dir.mkdir(exist_ok=True, parents=True)
        output_model_path = model_dir / "model"
        config.save(model_dir / "config.json")

        logger.info("reading dataset")
        dataset = df_to_dataset(dataset_df)

        logger.info("loading model and tokenizer")
        tokenizer = AutoTokenizer.from_pretrained(config.model_id)
        model = AutoModelForSequenceClassification.from_pretrained(config.model_id)
        if config.use_skip_connection:
            logger.info("using classification head with skip connection")
            model.classifier = SkipConnectionClassificationHead(model.config)
            model.post_init()
        model = model.to(config.device)

        logger.info("preprocessing dataset")
        preprocess_func = functools.partial(_preprocess_func, tokenizer=tokenizer)
        encoded_dataset = dataset.map(preprocess_func, batched=True)

        metric = evaluate.load(config.metric_name)
        compute_metrics = functools.partial(_compute_metrics, metric=metric)
        
        train_args = TrainingArguments(
            output_model_path,
            evaluation_strategy = "epoch",
            save_strategy = "epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=config.batch_size,
            per_device_eval_batch_size=config.batch_size,
            num_train_epochs=config.epochs,
            weight_decay=0.01,
            load_best_model_at_end=True,
            metric_for_best_model=config.metric_name,
            push_to_hub=False,
        )
        
        trainer = Trainer(
            model,
            train_args,
            train_dataset=encoded_dataset["train"],
            eval_dataset=encoded_dataset["dev"],
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )
        logger.info('starting training')
        trainer.train()

        # logger.info('evaluating model on dev after finetuning')
        # test_outputs = trainer.predict(encoded_dataset["dev"])
        # test_metrics = test_outputs.metrics
        # pprint(test_metrics)
