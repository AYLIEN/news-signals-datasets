import pandas as pd
import logging
from typing import List, Tuple, Callable


from ac.log import create_logger


logger = create_logger(__name__,  level=logging.INFO)


def remove_empty_samples(df):
    filter_func = lambda x: x["summary"]["summary"] is None
    df_new = df[~(df.apply(filter_func, axis=1) == True)]
    return df_new


def remove_empty_signals(dataset):
    for signal_id in sorted(dataset.signals):
        signal = dataset.signals[signal_id]
        days_with_stories = 0
        for stories in signal.feeds_df['stories']:
            if stories is not None and len(stories) > 0:
                days_with_stories += 1
        if days_with_stories == 0:
            logger.info(f'deleting signal with 0 stories: {signal.name}')
            del dataset.signals[signal_id]
    return dataset


def train_dev_test_splits(
    df,
    shuffle=True,
    random_state=42,
    train_ratio=0.6,
    dev_ratio=0.2,
):
    """
    Split the dataset into train, dev, test.
    """
    assert train_ratio + dev_ratio < 1, "train_ratio + dev_ratio must be < 1"    
    freq = pd.infer_freq(df.index) # TODO: returns None, fix this
    start = df.index.min()        
    end = df.index.max() + pd.Timedelta(days=1) # TODO: change to freq 
    unique_dates = pd.date_range(start, end, freq=freq)
    num_dates = len(unique_dates)
    train_start = unique_dates[0]
    train_end = unique_dates[int(num_dates * train_ratio)]
    dev_start = train_end
    dev_end = unique_dates[int(num_dates * (train_ratio + dev_ratio))]
    test_start = dev_end
    test_end = unique_dates[-1]
    df = df.copy()
    df["split_group"] = None
    df.loc[(df.index>=train_start) & (df.index < train_end), "split_group"] = "train"
    df.loc[(df.index>=dev_start) & (df.index < dev_end), "split_group"] = "dev"
    df.loc[(df.index>=test_start) & (df.index < test_end), "split_group"] = "test"
    if shuffle:
        df = df.sample(frac=1, random_state=random_state)
    return df


def discretize_anomaly_scores(df, threshold=1):
    func = lambda x: 1 if x["anomalies"] > threshold else 0
    labels = df.apply(func, result_type="expand", axis=1)
    df["is_anomaly"] = labels
    return df
    
    
def sample_balanced_labels(
    df,
    target_col="anomalies",
    label_counts=None,
    selected_split_groups=None
):
    if selected_split_groups is None:
        selected_split_groups = ["train"]
    dfs_by_split = []
    for split in df["split_group"].unique():
        # sample separately within each split
        split_df = df[df["split_group"] == split]
        if split not in selected_split_groups:
            dfs_by_split.append(split_df)
            continue
        dfs_by_label = []
        for label in split_df[target_col].unique():
            label_df = split_df[split_df[target_col] == label]
            # draw N samples of this label
            label_df = label_df.sample(
                n=label_counts[label], replace=True
            )
            dfs_by_label.append(label_df)
        # concat and shuffle for this split (e.g. train)
        split_df = pd.concat(dfs_by_label)
        split_df = split_df.sample(frac=1)
        dfs_by_split.append(split_df)
    df = pd.concat(dfs_by_split)
    return df


def remove_overlapping_rows(
    df_source: pd.DataFrame,
    df_target: pd.DataFrame,
    row_to_key: Callable
) -> pd.DataFrame:  
    """
    Examples from df_source will be removed from df_target.
    This is used to prevent leakage between training and dev/test data. 
    """
    forbidden_keys = set(df_source.apply(row_to_key, axis=1))
    filter_func = lambda x: row_to_key(x) in forbidden_keys
    df_target_clean = df_target[~(df_target.apply(filter_func, axis=1) == True)]
    return df_target_clean


def remove_leakage_across_splits(
    df: pd.DataFrame,
    removal_sequence: List[Tuple[str, str]],
    row_to_key: Callable
) -> pd.DataFrame:  
    """
    Given a sequence of <source, target> split pairs,
    e.g. [("test", "dev"), ("test", "train"), ("train", "dev")],
    we remove overlapping rows from the source in the target dataframe
    each time, in the provided order.
    """
    splits = df["split_group"].unique()
    split_to_df = dict(
        (split, df[df["split_group"] == split])
        for split in splits
    ) 
    for src_split, tgt_split in removal_sequence:
        df_tgt = remove_overlapping_rows(
            df_source=split_to_df[src_split],
            df_target=split_to_df[tgt_split],
            row_to_key=row_to_key
        )
        split_to_df[tgt_split] = df_tgt
    df = pd.concat([split_to_df[split] for split in splits])
    return df
