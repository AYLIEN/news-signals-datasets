VERSION       ?= `cat VERSION`

# Resources
RESOURCES_ROOT    := gs://aylien-science-files/aylien_timeseries
RESOURCES_VERSION ?= 1
RESOURCE_PATH     ?= resources/$(RESOURCES_VERSION)

TEST_RESOURCES_VERSION ?= test
TEST_STORAGE            = test_storage
STORAGE                ?= storage/$(TEST_STORAGE)

.PHONY: tag
tag:
	git tag -a $(VERSION) -m "version $(VERSION)"
	git push origin $(VERSION)

#########
## DEV ##
#########

.PHONY: dev
dev:
	pip install -r requirements.txt
	pip install -e .

###########
## TESTS ##
###########

.PHONY: test
test: $(resources-test)
	RESOURCES=resources/$(TEST_RESOURCES_VERSION) \
	python -W ignore -m unittest discover -p "test*.py"
	#flake8 aylien_timeseries --exclude schema_pb2.py


##########################
## DEV BUILD AND DEPLOY ##
##########################

# this is Aylien private repo, public users can
# modify args as needed
REGION          ?= europe-west1
PROJECT_ID      ?= aylien-science
REPOSITORY_NAME ?= aylien-science
IMAGE_NAME      ?= news-signals

TAG       ?= $(shell git describe --tags --dirty --always)
IMAGE_URI ?= $(REGION)-docker.pkg.dev/$(PROJECT_ID)/$(REPOSITORY_NAME)/$(IMAGE_NAME):$(TAG)

.PHONY: build
build:
	docker build -t $(IMAGE_URI) -f Dockerfile .

.PHONY: push
push:
	docker push $(IMAGE_URI)


.PHONY: container-test
container-test:
	mkdir -p sample_dataset_output/
	docker run \
	 	-u $(shell id -u):$(shell id -g) \
		-v $(shell pwd)/resources/dataset-config-example.json:/dataset-config-example.json \
		-e DATASET_CONFIG=resources/dataset-config-example.json \
		-v $(shell pwd)/sample_dataset_output:/sample_dataset_output \
		-e DATASET_CONFIG=/dataset-config-example.json \
		-e NEWSAPI_APP_ID=$(NEWSAPI_APP_ID) \
		-e NEWSAPI_APP_KEY=${NEWSAPI_APP_KEY} \
		$(IMAGE_URI)

################# 
# DOCUMENTATION #
################# 

# runs local mkdocs server on port 8000
.PHONY: docs-serve
docs-serve:
	mkdocs serve


######################
# DATASET GENERATION #
######################

DATASET_CONFIG ?= "resources/dataset-config-example.json"

.PHONY: create-dataset
create-dataset:
	python bin/generate_dataset.py --config $(DATASET_CONFIG)
