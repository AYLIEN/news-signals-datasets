from datasets import Dataset, DatasetDict, Features, Value, ClassLabel


def df_to_dataset(df):
    """
    TODO: If this varies, move to loader script.
    """
    features = Features({
        "text": Value("string"),
        "aspect": Value("string"), 
        "label": ClassLabel(num_classes=2, names=[0, 1]),    
    })
    dataset_dict = {}
    for split in df.split_group.unique():
        df_split = df[df["split_group"] == split]
        records = [
            {
                "text": x["summary"]["summary"],
                "label": x["is_anomaly"],
                "aspect": x["signal_name"],
            }
            for x in df_split.to_dict("records")
        ]
        dataset = Dataset.from_list(records)
        dataset.cast(features)
        dataset_dict[split] = dataset
    return DatasetDict(dataset_dict)