# Anomaly target: wikimedia_pageviews or counts
target_name=$1

dataset_names=(
nasdaq100.$target_name
smp500.$target_name
us-politicians.$target_name
)


for dataset_name in "${dataset_names[@]}"
do
    dataset_path="data/classification_datasets/dataset-"$dataset_name".tar.gz"
    model_path="data/models/"$dataset_name".rf_classifier"
    predictions_path="data/predictions/"$dataset_name".rf_classifier"

    python bin/train.py \
        --model-class SparseRandomForestClassifier \
        --config configs/sparse_rf_classifier.json \
        --dataset-path $dataset_path \
        --output-path data/models/"$dataset_name".rf_classifier

    python bin/predict_evaluate.py \
        --model-class SparseRandomForestClassifier \
        --config configs/sparse_rf_classifier.json \
        --dataset-path $dataset_path \
        --model-path $model_path \
        --output-path $predictions_path \
        --dataset-split test
done
