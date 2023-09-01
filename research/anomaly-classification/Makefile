RESOURCES_VERSION?=0.3


.PHONY: dev
dev:
	pip install -r requirements.txt
	pip install -e .


.PHONY: pull_signals_datasets
pull_signals_datasets:
	mkdir -p data/datasets
	gsutil -m cp -r gs://aylien-science-files/newssignals-dataset/resources/$(RESOURCES_VERSION)/datasets/* data/datasets


.PHONY: pull_classification_datasets
pull_classification_datasets:
	mkdir -p data/classification_datasets
	gsutil -m cp -r gs://aylien-science-files/newssignals-dataset/resources/$(RESOURCES_VERSION)/classification-datasets/* data/classification_datasets


# target can be: count or wikimedia_pageviews
TARGET_NAME ?= count

.PHONY: create_classification_datasets
create_classification_datasets:
	mkdir -p data/classification_datasets
	bash bin/create_classification_datasets.sh $(TARGET_NAME)


.PHONY: sparse_experiments
sparse_experiments:
	bash bin/run_sparse_experiments.sh count
	bash bin/run_sparse_experiments.sh wikimedia_pageviews


TRANSFORMER_CONFIG ?= configs/transformer_classifier.json
TRANSFORMER_OUTPUT_SUFFIX ?= transformer_classifier


.PHONY: transformer_experiments
transformer_experiments:
	bash bin/run_transformer_experiments.sh $(TARGET_NAME) $(TRANSFORMER_CONFIG) $(TRANSFORMER_OUTPUT_SUFFIX)


.PHONY: random_experiments
random_experiments:
	bash bin/run_random_experiments.sh count configs/random-target.json random-target
	bash bin/run_random_experiments.sh count configs/random-uniform.json random-uniform
	bash bin/run_random_experiments.sh wikimedia_pageviews configs/random-target.json random-target
	bash bin/run_random_experiments.sh wikimedia_pageviews configs/random-uniform.json random-uniform


LLM_CONFIG = configs/llm_classifier.json
LLM_OUTPUT_SUFFIX ?= llm_classifier


.PHONY: llm_experiments
llm_experiments:
	bash bin/run_llm_experiments.sh $(TARGET_NAME) $(LLM_CONFIG) $(LLM_OUTPUT_SUFFIX)

.PHONY: llm_wiki_experiments
llm_wiki_experiments:
	bash bin/run_llm_experiments.sh wikimedia_pageviews configs/llama_classifier.prompt-generic-1.json llm_classifier.prompt-wiki-1

.PHONY: llm_count_experiments
llm_count_experiments:
	bash bin/run_llm_experiments.sh count configs/llama_classifier.prompt-generic-1.json llm_classifier.prompt-generic-1

.PHONY: try_llama
try_llama:
	python bin/predict_evaluate.py \
		--model-class LLMClassifier \
		--config configs/llm_classifier.json \
		--dataset-path data/classification_datasets/dataset-us-politicians.wikimedia_pageviews.tar.gz \
		--output-path data/predictions/us-politicians.wikimedia_pageviews.llm \
		--dataset-split test \
		--first-k 100