FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt -y update \
    && apt -y install \
        bsdutils \
        build-essential \
        curl \
        ffmpeg \
        libsm6 \
        libxext6 \
        libpq-dev \
        python3-venv \
        python3-pip \
    && apt upgrade -y \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/app

RUN conda update conda \
    && conda config --add channels conda-forge \
    && conda config --set channel_priority strict

RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH="/opt/miniconda/envs/env/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

RUN pip install --upgrade setuptools \
    && pip3 install --upgrade setuptools

# Initialize Conda, activate environment and install poetry packages
RUN /opt/miniconda/bin/conda init bash && \
    . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate env && \
    poetry update --no-ansi --no-interaction && \
    poetry install --no-ansi --no-interaction --no-root
