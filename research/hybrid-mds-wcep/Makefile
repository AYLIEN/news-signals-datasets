GS_MODEL_LOCATION ?= gs://aylien-science-files/demian/hybrid-mds-wcep/models/ext-bart-large/checkpoint-20000
MODEL_PATH ?= models/ext-bart-large
CHECKPOINT_PATH ?= models/ext-bart-large/test-checkpoint-20000
# TRAIN_DATA_PATH ?= <insert-train-data>


.PHONY: dev
dev:
	pip install -e .
	python -m spacy download en_core_web_sm


.PHONY: download_model
download_model:
	mkdir -p $(CHECKPOINT_PATH)
	gsutil cp -r $(GS_MODEL_LOCATION) $(CHECKPOINT_PATH)


.PHONY: example
example:
	python examples/example.py


# TODO: enable original training here
# TODO: clone transformers repo 
# .PHONY: finetune
# finetune:
# 	python examples/pytorch/summarization/run_summarization.py \
# 		--model_name_or_path facebook/bart-large \
# 		--do_train \
# 		--train_file $(TRAIN_DATA_PATH) \
# 		--output_dir $(MODEL_PATH) \
# 		--num_train_epochs=5 \
# 		--per_device_train_batch_size=2 \
# 		--per_device_eval_batch_size=2 \
# 		--overwrite_output_dir \
# 		--predict_with_generate


