# Anomaly target: wikimedia_pageviews or counts
target_name=$1
config_path=$2
output_suffix=$3

dataset_names=(
us-politicians.$target_name
nasdaq100.$target_name
smp500.$target_name
)

for dataset_name in "${dataset_names[@]}"
do
    dataset_path="data/classification_datasets/dataset-$dataset_name.tar.gz"
    predictions_path="data/predictions/$dataset_name.$output_suffix"

    python bin/predict_evaluate.py \
        --model-class RandomClassifier \
        --config $config_path \
        --dataset-path $dataset_path \
        --output-path $predictions_path \
        --dataset-split test
done
