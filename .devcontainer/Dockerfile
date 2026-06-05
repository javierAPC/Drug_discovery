FROM mcr.microsoft.com/devcontainers/python:3.10

RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    swig \
    wget \
    && apt-get clean


COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
