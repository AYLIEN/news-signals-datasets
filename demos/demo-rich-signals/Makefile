# Demo port
PORT ?= 9001

############
# BUILDING #
############

.PHONY: dev
dev:
	pip install -r requirements.txt

#########
## RUN ##
#########

# Tasks
.PHONY: run
run:
	streamlit run \
		--server.port $(PORT) \
		--server.baseUrlPath rich-signals \
		rich_signals_demo.py
