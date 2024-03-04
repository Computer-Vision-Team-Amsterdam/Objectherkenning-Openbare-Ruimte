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
RUN echo "source activate env" > ~/.bashrc
ENV PATH="/opt/miniconda/envs/env/bin:$PATH"
RUN conda install -c conda-forge conda-pack

RUN pip install pipx \
    && pipx ensurepath
RUN pipx install poetry
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment and install poetry packages
RUN /opt/miniconda/bin/conda init bash \
    && . /opt/miniconda/etc/profile.d/conda.sh \
    && conda activate env \
    && poetry update --no-ansi --no-interaction \
    && poetry install --no-ansi --no-interaction --no-root

# Use conda-pack to create a standalone env in /venv
RUN conda-pack -n env -o /tmp/env.tar && \
    mkdir /venv && cd /venv && tar xf /tmp/env.tar && \
    rm /tmp/env.tar

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
COPY --from=builder /venv /venv

#COPY model_artifacts/dataoffice_model/last-purple_boot_3l6p24vb.pt model_artifacts/last-purple_boot_3l6p24vb.pt
COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG AML_MODEL_ID_arg
ENV AML_MODEL_ID=$AML_MODEL_ID_arg
ARG PROJECT_VERSION_arg
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]