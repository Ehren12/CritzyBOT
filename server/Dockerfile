# https://hub.docker.com/_/python
FROM python:3.10-slim-bullseye

ENV PYTHONUNBUFFERED True
ENV APP_HOME /app
WORKDIR $APP_HOME
COPY . ./

RUN apt update
RUN apt install -y pkg-config
RUN apt install -y libcairo2-dev python3-dev libcairo2
RUN apt install -y gcc
RUN apt install -y build-essential
RUN apt install -y libgirepository1.0-dev

RUN pip install --no-cache-dir -r ./server/requirements.txt

CMD ["uvicorn", "server.main:app", "--host", "0.0.0.0", "--port", "8080"]
