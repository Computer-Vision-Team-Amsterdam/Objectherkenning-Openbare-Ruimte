FROM python:3.8.12-slim AS base-image

# Upgrade and install system libraries
RUN apt-get -y update \
    && ACCEPT_EULA=Y apt-get upgrade -qq \
    && apt-get -y install \
        build-essential \
        curl \
        git \
        yasm \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /usr/src
RUN git clone https://git.ffmpeg.org/ffmpeg.git
RUN cd ffmpeg \
    && ./configure \
    && make \
    && make install

RUN pip install pipx \
    && pipx ensurepath
RUN pipx install poetry
ENV PATH=/root/.local/bin:$PATH

RUN poetry config virtualenvs.create false

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment and install poetry packages
RUN poetry update --no-ansi --no-interaction && \
    poetry install --no-ansi --no-interaction --no-root

#COPY model_artifacts/dataoffice_model/last-purple_boot_3l6p24vb.pt model_artifacts/last-purple_boot_3l6p24vb.pt
COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG AML_MODEL_ID_arg
ENV AML_MODEL_ID=$AML_MODEL_ID_arg
ARG PROJECT_VERSION_arg
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
