import json
from collections import Counter

import pandas as pd
from sklearn.metrics import precision_recall_fscore_support

from ac.data_utils import df_to_dataset
    

def measure_prf(truth, preds, positive_label):
    P, R, F, _ = precision_recall_fscore_support(truth, preds, labels=[positive_label])
    p, r, f = P[0], R[0], F[0]
    return p, r, f


def get_label_ratio(labels):
    counts = Counter(labels)
    n = len(labels)
    positive_count = counts[1]
    positive_ratio = 0 if positive_count == 0 else positive_count / n
    return positive_ratio


def evaluate(predictions, examples): 
    # Note this is aspect-based text classification, i.e.
    # each example is a {"text": "...", "aspect": "..."} item.

    y_true = [x["label"] for x in examples]
    y_pred = [x["predicted_label"] for x in predictions]
    
    assert len(y_true) == len(y_pred)
    p, r, f = measure_prf(y_true, y_pred, positive_label=1)    
    true_positive_ratio = get_label_ratio(y_true)
    pred_positive_ratio = get_label_ratio(y_pred)

    print("precision:", p)
    print("recall:   ", r)
    print("f1:       ", f)
    print("actual positive ratio   :", round(true_positive_ratio, 3))
    print("predicted positive ratio:", round(pred_positive_ratio, 3))

    results = {
        "precision": p,
        "recall": r,
        "f1": f,
        "true_positive_ratio": true_positive_ratio,
        "pred_positive_ratio": pred_positive_ratio
    }
    return results
