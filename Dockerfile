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

RUN git clone https://git.ffmpeg.org/ffmpeg.git
RUN cd ffmpeg \
    && ./configure \
    && make \
    && make install

ENV PATH=/root/.local/bin:$PATH

RUN pip3 install pipx \
    && pipx ensurepath \

RUN pipx install poetry \
    && pipx inject poetry poetry-plugin-export

WORKDIR /usr/src

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry export --without-hashes --format=requirements.txt > requirements.txt \
    && pip3 install -r requirements.txt

COPY model_artifacts/oor_model/yolov8s_coco_nc_5_best.pt model_artifacts/yolov8s_coco_nc_5_best.pt
COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG ML_MODEL_ID_arg=1
ENV ML_MODEL_ID=$ML_MODEL_ID_arg
ARG PROJECT_VERSION_arg=1
ENV PROJECT_VERSION=$PROJECT_VERSION_arg
ARG SHARED_ACCESS_KEY_IOT_arg=1
ENV SHARED_ACCESS_KEY_IOT=$SHARED_ACCESS_KEY_IOT_arg
ARG AI_INSTRUMENTATION_KEY_arg=1
ENV AI_INSTRUMENTATION_KEY=$AI_INSTRUMENTATION_KEY_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
