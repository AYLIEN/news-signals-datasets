# Anomaly target: wikimedia_pageviews or counts
target_name=$1

dataset_names=(
us-politicians.$target_name
nasdaq100.$target_name
smp500.$target_name
)

config_path=$2
output_suffix=$3

for dataset_name in "${dataset_names[@]}"
do
    dataset_path="data/classification_datasets/dataset-"$dataset_name".tar.gz"
    model_path="data/models/"$dataset_name".transformer_classifier"
    predictions_path="data/predictions/$dataset_name.$output_suffix"

    python bin/train.py \
        --model-class TransformerClassifier \
        --config $config_path \
        --dataset-path $dataset_path \
        --output-path data/models/$dataset_name.$output_suffix 

    python bin/predict_evaluate.py \
        --model-class TransformerClassifier \
        --config $config_path \
        --dataset-path $dataset_path \
        --model-path $model_path \
        --output-path $predictions_path \
        --dataset-split test
done
