FROM mcr.microsoft.com/devcontainers/python:3.10

# Instalar dependencias de sistema para RDKit
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    swig \
    wget \
    && apt-get clean

# Instalar librerías Python
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt
