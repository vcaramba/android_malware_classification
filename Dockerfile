FROM continuumio/miniconda3:4.3.27

RUN mkdir -p /usr/src/app
WORKDIR /usr/src/app

COPY . /usr/src/app/

RUN apt-get update \
    && apt-get clean \
    && apt-get update -qqq \
    && apt-get install -y -q g++ \
    && conda install python=3.7.4 \
    && pip install --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

CMD jupyter notebook --port=8888 --no-browser --ip=0.0.0.0 --allow-root