#FROM --platform=linux/arm64/v8 nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS builder
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS builder

# Upgrade and install system libraries
RUN apt-get -y update \
    && apt-get upgrade -y --fix-missing \
    && apt-get install -y --no-install-recommends \
        wget \
        build-essential \
        curl \
        gcc \
        zip \
        htop \
        libgl1 \
        libglib2.0-0 \
        libpython3-dev \
        gnupg \
        g++ \
        libusb-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


WORKDIR /usr/src

ENV PATH /opt/conda/bin:$PATH

CMD [ "/bin/bash" ]

# Leave these args here to better use the Docker build cache
ARG CONDA_VERSION=latest

RUN set -x && \
    UNAME_M="$(uname -m)" && \
    if [ "${UNAME_M}" = "x86_64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-x86_64.sh"; \
        SHA256SUM="3f2e5498e550a6437f15d9cc8020d52742d0ba70976ee8fce4f0daefa3992d2e"; \
    elif [ "${UNAME_M}" = "s390x" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-s390x.sh"; \
        SHA256SUM="0489909051fd9e2c9addfa5fbd531ccb7f8f2463ac47376b8854e5a09b1c4011"; \
    elif [ "${UNAME_M}" = "aarch64" ]; then \
        MINICONDA_URL="https://repo.anaconda.com/miniconda/Miniconda3-${CONDA_VERSION}-Linux-aarch64.sh"; \
        SHA256SUM="1e046ef2d9d47289db2491f103c81b0b4baf943a9234ac59bd5bca076c881d98"; \
    fi && \
    wget "${MINICONDA_URL}" -O miniconda.sh -q && \
    echo "${SHA256SUM} miniconda.sh" > shasum && \
    if [ "${CONDA_VERSION}" != "latest" ]; then sha256sum --check --status shasum; fi && \
    mkdir -p /opt && \
    bash miniconda.sh -b -p /opt/conda && \
    rm miniconda.sh shasum && \
    ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc && \
    echo "conda activate base" >> ~/.bashrc && \
    find /opt/conda/ -follow -type f -name '*.a' -delete && \
    find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
    /opt/conda/bin/conda clean -afy

RUN conda create -n env python=3.8
RUN conda install -c conda-forge conda-pack
ENV PATH="/opt/conda/envs/env/bin:$PATH"

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
RUN /opt/conda/bin/conda init bash \
    && . /opt/conda/etc/profile.d/conda.sh \
    && conda activate env \
    && poetry update --no-ansi --no-interaction \
    && poetry install --no-ansi --no-interaction --no-root

WORKDIR /venv

# Use conda-pack to create a standalone env in /venv
RUN conda-pack -n env -o /venv/env.tar.gz --ignore-missing-files

#FROM --platform=linux/arm64/v8 nvidia/cuda:12.3.2-cudnn9-runtime-ubuntu22.04 AS runtime
FROM mcr.microsoft.com/azureml/openmpi4.1.0-cuda11.8-cudnn8-ubuntu22.04 AS runtime

RUN apt-get -y update \
  && apt-get upgrade -y --fix-missing \
  && echo "Europe/Amsterdam" > /etc/timezone \
  && DEBIAN_FRONTEND=noninteractive \
  && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        git \
        yasm \
        gcc \
        zip \
        htop \
        libgl1 \
        libglib2.0-0 \
        libpython3-dev \
        gnupg \
        g++ \
        libusb-1.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


RUN git clone https://git.ffmpeg.org/ffmpeg.git
RUN cd ffmpeg \
    && ./configure \
    && make \
    && make install

# Copy /venv from build stage
WORKDIR /venv
COPY --from=builder /venv .
COPY --from=builder /etc/ssl/certs/ca-certificates.crt /etc/ssl/certs/
RUN tar -xzf env.tar.gz

# Downloads to user config dir
ADD https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.ttf \
    https://github.com/ultralytics/assets/releases/download/v0.0.0/Arial.Unicode.ttf \
    /root/.config/Ultralytics/

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

