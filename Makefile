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


############################################
# Datascience Project Quickstarter Tooling #
############################################

# see https://github.com/AYLIEN/datascience-project-quickstarter for more info
# on these Makefile commands
PROJECT_DIR ?= new-project
PKG_NAME ?= new_pkg
PORT ?= 8000
CONTAINER := news_signals
VERSION ?= `cat VERSION`
DEMO_NAME ?= new-demo

# initialize a new project
.PHONY: new-project
new-project:
	python templates/create_project.py \
		--project-dir $(PROJECT_DIR) \
		--pkg-name $(PKG_NAME)


# initialize a new demo within current project
.PHONY: new-demo
new-demo:
	python templates/create_demo.py --dirname $(DEMO_NAME)
	echo "Finished creating new demo: $(DEMO_NAME)"
	echo "To run, do: cd demos/$(DEMO_NAME) && make run"
