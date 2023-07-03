FROM python:3.10-bullseye

WORKDIR /srv

RUN pip install --upgrade pip

# files and packages required for installation
ADD news_signals ./news_signals
ADD requirements.txt ./
ADD resources ./resources
ADD bin ./bin
ADD setup.py Makefile VERSION README.md ./

RUN make dev

CMD make create-dataset