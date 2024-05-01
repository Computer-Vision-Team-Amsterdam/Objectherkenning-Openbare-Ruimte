FROM ubuntu:22.04

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"

RUN apt-get update && apt-get install -y --no-install-recommends \
    tzdata \
    python3.10 \
    python3-pip \
    python3.10-venv \
    nano \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install pipx \
    && pipx ensurepath
RUN pipx install poetry
ENV PATH=/root/.local/bin:$PATH
RUN poetry config virtualenvs.create false

ENV CURLOPT_SSL_VERIFYHOST=0
ENV CURLOPT_SSL_VERIFYPEER=0

COPY pyproject.toml .
COPY poetry.lock .

# Initialize Conda, activate environment and install poetry packages
RUN poetry update --no-ansi --no-interaction \
    && poetry install --no-ansi --no-interaction --no-root

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
