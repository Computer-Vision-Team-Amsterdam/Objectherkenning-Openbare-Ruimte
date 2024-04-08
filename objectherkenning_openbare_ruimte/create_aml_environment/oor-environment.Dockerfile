FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS base-image

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

RUN conda create -n env python=3.8
RUN echo "source activate env" > ~/.bashrc
ENV PATH="/opt/miniconda/envs/env/bin:$PATH"

RUN curl -sSL https://install.python-poetry.org | python3 -
ENV PATH="/root/.local/bin:$PATH"
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment
RUN /opt/miniconda/bin/conda init bash && \
    . /opt/miniconda/etc/profile.d/conda.sh && \
    conda activate env && \
    apt-get update && \
    apt install -y libgl1-mesa-glx

# Install PyTorch, torchvision, and torchaudio with CUDA support
RUN pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu118

# Install other dependencies with Poetry
RUN poetry install --no-ansi --no-interaction --no-root

# Initialize Conda, activate environment and install poetry packages
# RUN /opt/miniconda/bin/conda init bash && \
#    . /opt/miniconda/etc/profile.d/conda.sh && \
#    conda activate env && \
#    apt-get update && \
#    apt install -y libgl1-mesa-glx && \
#    poetry update --no-ansi --no-interaction && \
#    poetry install --no-ansi --no-interaction --no-root