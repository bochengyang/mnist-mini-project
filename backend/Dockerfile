FROM python:3.9.6-slim

COPY ./requirements.txt /app/requirements.txt

WORKDIR /app

RUN pip install -r requirements.txt

ADD . /app

ENTRYPOINT ["/app/docker-entrypoint.sh"]