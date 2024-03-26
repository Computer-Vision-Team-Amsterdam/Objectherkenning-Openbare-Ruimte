FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS builder

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src

RUN conda create -n env python=3.8
RUN conda install -c conda-forge conda-pack
ENV PATH="/opt/miniconda/envs/env/bin:$PATH"

RUN pip install pipx \
    && pipx ensurepath
RUN pipx install poetry
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

ENV CURLOPT_SSL_VERIFYHOST=0
ENV CURLOPT_SSL_VERIFYPEER=0

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment and install poetry packages
RUN /opt/miniconda/bin/conda init bash \
    && . /opt/miniconda/etc/profile.d/conda.sh \
    && conda activate env \
    && poetry update --no-ansi --no-interaction \
    && poetry install --no-ansi --no-interaction --no-root

WORKDIR /venv

# Use conda-pack to create a standalone env in /venv
RUN conda-pack -n env -o /venv/env.tar.gz --ignore-missing-files

FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS runtime

RUN apt-get -y update \
  && apt-get upgrade -y --fix-missing \
  && echo "Europe/Amsterdam" > /etc/timezone \
  && DEBIAN_FRONTEND=noninteractive \
  && apt-get -y install \
        build-essential \
        curl \
        git \
        yasm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone https://git.ffmpeg.org/ffmpeg.git \
    && cd ffmpeg \
    && ./configure \
    && make \
    && make install

# Copy /venv from build stage
WORKDIR /venv
COPY --from=builder /venv .
RUN tar -xzf env.tar.gz

WORKDIR /usr/src
# This needs to be replaced to a generic model name when it's actually deployed
#COPY model_artifacts/dataoffice_model/last-purple_boot_3l6p24vb.pt model_artifacts/last-purple_boot_3l6p24vb.pt
COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG AML_MODEL_ID_arg=1
ENV AML_MODEL_ID=$AML_MODEL_ID_arg
ARG PROJECT_VERSION_arg=1
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
