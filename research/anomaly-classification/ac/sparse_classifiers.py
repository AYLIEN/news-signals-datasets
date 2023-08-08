import logging
import joblib
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import classification_report
from ac.config import Config
from ac.classifier import Classifier
from ac.utils import load_variable
from ac.log import create_logger


logger = create_logger(__name__,  level=logging.INFO)



def example_to_text(x):
    return f"{x['aspect']} {x['text']}"


def dataset_to_text_and_labels(dataset_df, split):
    split_data = dataset_df[dataset_df["split_group"] == split]
    examples = [
        {
            "text": x["summary"]["summary"],
            "label": x["is_anomaly"],
            "aspect": x["signal_name"],
        }
        for x in split_data.to_dict("records")
    ]        
    texts = [example_to_text(x) for x in examples]
    labels = [x["label"] for x in examples]
    return texts, labels
    

class SparseLinearClassifierConfig(Config):
    def __init__(self, args, overwriting_args=None):
        super().__init__()        
        self.register_param('vectorizer', str, possible_values=('CountVectorizer', 'TfidfVectorizer'))
        self.register_param('max_features', int, 5000)
        self.set_params_from_args(args, overwriting_args)


class SparseLinearClassifier(Classifier):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
            
    @classmethod
    def load(cls, config, path):
        """
        Loads classifier from model directory.
        """        
        model_dir = Path(path)
        model = joblib.load(model_dir / "model.pkl")
        vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        return cls(model, vectorizer)

    @classmethod
    def train(
        cls,
        config: SparseLinearClassifierConfig,
        dataset_df,
        output_path
    ):
        model_dir = Path(output_path)
        model_dir.mkdir(exist_ok=True)
        
        config.save(model_dir / "config.yaml")        
        logger.info("preprocessing dataset")
        train_texts, Y_train = dataset_to_text_and_labels(dataset_df, "train")
        dev_texts, Y_dev = dataset_to_text_and_labels(dataset_df, "dev")

        if config.vectorizer == 'TfidfVectorizer':
            vectorizer = TfidfVectorizer(
                 stop_words='english',
                 lowercase=True,
                 max_features=config.max_features
            )
        else:
             vectorizer = CountVectorizer(
                 stop_words='english',
                 lowercase=True,
                 max_features=config.max_features,
                 binary=True
             )

        model = LogisticRegression()

        X_train = vectorizer.fit_transform(train_texts)
        X_dev = vectorizer.transform(dev_texts)

        logger.info("training model")
        model = model.fit(X_train, Y_train)

        P_train = model.predict(X_train)
        P_dev = model.predict(X_dev)

        logger.info("Eval on training data:")
        logger.info("\n" + str(classification_report(Y_train, P_train)))
        logger.info("Eval on dev data:")
        logger.info("\n" + str(classification_report(Y_dev, P_dev)))

        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")    

    def predict(self, examples, **kwargs):
        texts = [example_to_text(x) for x in examples]
        X = self.vectorizer.transform(texts)
        predicted_labels = list(self.model.predict(X))
        return [{'predicted_label': int(y)} for y in predicted_labels]


class SparseRandomForestClassifierConfig(Config):
    def __init__(self, args, overwriting_args=None):
        super().__init__()        
        self.register_param('vectorizer', str, possible_values=('CountVectorizer', 'TfidfVectorizer'))
        self.register_param('max_features', int, 5000)
        self.register_param('n_estimators', int, 10)
        self.register_param('max_depth', int, 5)
        self.set_params_from_args(args, overwriting_args)


class SparseRandomForestClassifier(Classifier):
    def __init__(self, model, vectorizer):
        self.model = model
        self.vectorizer = vectorizer
            
    @classmethod
    def load(cls, config, path):
        """
        Loads classifier from model directory.
        """        
        model_dir = Path(path)
        model = joblib.load(model_dir / "model.pkl")
        vectorizer = joblib.load(model_dir / "vectorizer.pkl")
        return cls(model, vectorizer)

    @classmethod
    def train(
        cls,
        config: SparseRandomForestClassifierConfig,
        dataset_df,
        output_path
    ):
        model_dir = Path(output_path)
        model_dir.mkdir(exist_ok=True)
        
        config.save(model_dir / "config.json")
        logger.info("preprocessing dataset")

        train_texts, Y_train = dataset_to_text_and_labels(dataset_df, "train")
        dev_texts, Y_dev = dataset_to_text_and_labels(dataset_df, "dev")

        if config.vectorizer == 'TfidfVectorizer':
            vectorizer = TfidfVectorizer(
                 stop_words='english',
                 lowercase=True,
                 max_features=config.max_features
            )
        else:
             vectorizer = CountVectorizer(
                 stop_words='english',
                 lowercase=True,
                 max_features=config.max_features,
                 binary=True
             )

        model = RandomForestClassifier(
            n_estimators=config.n_estimators,
            max_depth=config.max_depth
        )

        X_train = vectorizer.fit_transform(train_texts)
        X_dev = vectorizer.transform(dev_texts)

        logger.info("training model")
        model = model.fit(X_train, Y_train)

        P_train = model.predict(X_train)
        P_dev = model.predict(X_dev)

        logger.info("Eval on training data:")
        logger.info("\n" + str(classification_report(Y_train, P_train)))
        logger.info("Eval on dev data:")
        logger.info("\n" + str(classification_report(Y_dev, P_dev)))

        joblib.dump(model, model_dir / "model.pkl")
        joblib.dump(vectorizer, model_dir / "vectorizer.pkl")    

    def predict(self, examples, **kwargs):
        texts = [example_to_text(x) for x in examples]
        X = self.vectorizer.transform(texts)
        predicted_labels = list(self.model.predict(X))
        return [{'predicted_label': int(y)} for y in predicted_labels]
