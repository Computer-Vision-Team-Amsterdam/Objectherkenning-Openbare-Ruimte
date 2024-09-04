FROM ultralytics/ultralytics:latest-jetson-jetpack5

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"

RUN apt-get update  \
    && apt-get upgrade -y --fix-missing \
    && apt-get install -y --no-install-recommends \
    tzdata \
    python3-pip \
    python3-venv \
    nano \
    python3-dev \
    build-essential \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

ENV PATH=/root/.local/bin:$PATH

RUN python3 -m pip install --upgrade pip

WORKDIR /usr/src

COPY requirements_on_edge.txt .

# Need to install this separately to prevent CVToolkit deps from installing Torch
RUN python3 -m pip install --no-deps git+https://github.com/Computer-Vision-Team-Amsterdam/CVToolkit.git

RUN python3 -m pip install -r requirements_on_edge.txt

COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG ML_MODEL_ID_arg=1
ENV ML_MODEL_ID=$ML_MODEL_ID_arg
ARG PROJECT_VERSION_arg=1
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
