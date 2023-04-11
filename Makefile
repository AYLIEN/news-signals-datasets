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
	pip --use-deprecated=legacy-resolver install -e .
	pip install -r requirements.txt

###########
## TESTS ##
###########

.PHONY: test
test: $(resources-test)
	RESOURCES=resources/$(TEST_RESOURCES_VERSION) \
	python -W ignore -m unittest discover -p "test*.py"
	#flake8 aylien_timeseries --exclude schema_pb2.py


# build docker container
.PHONY: build
build:
	docker build --no-cache -t $(CONTAINER):$(VERSION) -f Dockerfile .

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

DATASET_CONFIG ?= "resources/sample-dataset-config.json"

.PHONY: create-dataset
create-dataset:
	python bin/generate_dataset.py --config $(DATASET_CONFIG)
