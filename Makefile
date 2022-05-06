PROJECT_DIR ?= new-project
PKG_NAME ?= new_pkg
PORT ?= 8000
CONTAINER := aylien_ts_datasets
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

.PHONY: dev
dev:
	pip install -e .


.PHONY: test
test:
	python -Wignore -m unittest discover ; \
	flake8 aylien_ts_datasets bin --exclude schema_pb2.py ; \
	#black aylien_ts_datasets bin --check --line-length=79 --exclude schema_pb2.py --experimental-string-processing


.PHONY: proto
proto:
	protoc --python_out=aylien_ts_datasets schema.proto


# run service locally
.PHONY: run
run:
	python -m aylien_model_serving \
		--handler aylien_ts_datasets.serving \
		--port $(PORT)


# build docker container
.PHONY: build
build:
	docker build --no-cache -t $(CONTAINER):$(VERSION) -f Dockerfile .
