FROM nvcr.io/nvidia/l4t-pytorch:r35.2.1-pth2.0-py3

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ="Europe/Amsterdam"

RUN apt-get update  \
    && apt-get upgrade -y --fix-missing \
    && apt-get install -y --no-install-recommends \
    tzdata \
    python3-pip \
    python3-venv \
    nano \
    gcc \
    python3-dev \
    git \
    build-essential \
    htop \
    libgl1 \
    libglib2.0-0 \
    libpython3-dev \
    gnupg \
    g++ \
    libusb-1.0-0 \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN git clone https://git.ffmpeg.org/ffmpeg.git
RUN cd ffmpeg \
    && ./configure \
    && make \
    && make install

RUN pip3 install pipx \
    && pipx ensurepath
RUN pipx install poetry
ENV PATH=/root/.local/bin:$PATH

COPY pyproject.toml .
COPY poetry.lock .

RUN poetry update --no-ansi --no-interaction --no-dev \
    && poetry install --no-ansi --no-interaction --no-dev --no-root

ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

WORKDIR /usr/src
COPY model_artifacts/oor_model/yolov8s_coco_nc_5_best.pt model_artifacts/yolov8s_coco_nc_5_best.pt
COPY objectherkenning_openbare_ruimte objectherkenning_openbare_ruimte
COPY config.yml config.yml

ARG AML_MODEL_ID_arg=1
ENV AML_MODEL_ID=$AML_MODEL_ID_arg
ARG PROJECT_VERSION_arg=1
ENV PROJECT_VERSION=$PROJECT_VERSION_arg

COPY entrypoint.sh .

ENTRYPOINT ["/bin/bash", "/usr/src/entrypoint.sh"]
