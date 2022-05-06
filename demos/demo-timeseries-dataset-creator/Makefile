DEMO_NAME     ?= basename `pwd`
CONTAINER     ?= `echo $(DEMO_NAME) | tr "_" "-"`-demo
VERSION       ?= `cat VERSION`
PORT		  ?= 8000


.PHONY: run
run:
	streamlit --log_level debug run \
			  --server.port $(PORT) \
			  --server.baseUrlPath $(DEMO_NAME) \
		  	  demo.py


.PHONY: build
build:
	cd ../..; docker build --no-cache -t $(CONTAINER):$(VERSION) -f demos/demo-timeseries-dataset-creator/Dockerfile .
