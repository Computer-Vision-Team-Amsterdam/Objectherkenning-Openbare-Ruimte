FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04:20240614.v1 AS base-image

# Upgrade and install system libraries
RUN apt-get -y update \
    && apt-get -y install \
        bsdutils \
        build-essential \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libpq-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

RUN conda update conda \
    && conda config --add channels conda-forge \
    && conda config --set channel_priority strict

RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH="/opt/miniconda/envs/env/bin:$PATH"

# EVERYTHING UNTIL HERE without vulnerabilities
# Somewhere in the lines below "setuptools 56.0.0.0" is installed, but no evidence of this in the build log

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment and install poetry packages
RUN /opt/miniconda/bin/conda init bash && \
    . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate env && \
    poetry update --no-ansi --no-interaction && \
    poetry install --no-ansi --no-interaction --no-root
