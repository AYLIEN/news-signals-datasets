FROM python:3.7.9-stretch

WORKDIR /srv

RUN ls .

RUN pip install --upgrade pip

ADD aylien_ts_datasets ./aylien_ts_datasets
ADD requirements.txt ./
ADD VERSION ./
ADD setup.py ./
ADD demos/demo-timeseries-dataset-creator/demo.py ./
ADD demos/demo-timeseries-dataset-creator/Makefile ./

RUN ls .

RUN pip install -e .

CMD make run
