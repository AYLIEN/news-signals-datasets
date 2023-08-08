#!/bin/bash

# List of inputs
inputs=(
"data/datasets/dataset-nasdaq100.tar.gz"
"data/datasets/dataset-smp500.tar.gz"
"data/datasets/dataset-us-politicians.tar.gz"
)


# Anomaly target: wikimedia_pageviews or counts
target_name=$1

# Config path
config="configs/classification_dataset.$target_name.json"

echo $target_name
echo $config

# Directory for classification datasets
output_dir="data/classification_datasets"

# Python script path
script_path="bin/create_classification_dataset.py"

for input in "${inputs[@]}"
do
    # Extract dataset name
    dataset_name=$(basename "$input" .tar.gz)

    # Construct output path
    output="$output_dir/$dataset_name.$target_name.tar.gz"

    python $script_path --input-dataset-path $input --output-dataset-path $output --config $config
done
