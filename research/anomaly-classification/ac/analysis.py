import json
from pathlib import Path
import pandas as pd


def results_to_dataframe(predictions_dir) -> pd.DataFrame:
    predictions_dir = Path(predictions_dir)
    prediction_items = []
    for pred_dir in predictions_dir.iterdir():
        with open(pred_dir / "config.json") as f:
            config = json.load(f)
        results_path = Path(pred_dir / "evaluation.json")
        if not results_path.exists():
            continue
        results = json.loads(results_path.read_text())
        item = {
            "config": config,
            "results": results,
            "name": pred_dir.name
        }
        prediction_items.append(item)
    
    rows = []
    for item in  prediction_items:
        row = {
            "name": item["name"],
        }
        row.update(item["config"])
        row.update(item["results"])
        rows.append(row)
    df = pd.DataFrame(data=rows)
    return df
