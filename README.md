# Objectherkenning-Openbare-Ruimte
This project is about recognising objects from public space images.


## Installation

#### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/Objectherkenning-Openbare-Ruimte.git
```

#### 2. Install Poetry
If you don't have it yet, follow the instructions [here](https://python-poetry.org/docs/#installation) to install the package manager Poetry.


#### 3. Init submodules
You need to initialize the content of the submodules so git clones the latest version.
```bash
git submodule update --init --recursive
```

#### 4. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use Poetry to create a new virtual environment and install the dependencies.

```bash
poetry install
```
    
#### 5. Install pre-commit hooks
The pre-commit hooks help to ensure that all committed code is valid and consistently formatted.

```bash
poetry run pre-commit install
```

### Manually upload docker image to device

#### On the laptop:
Install builder (if building on an AMD64 architecture):

```bash
docker buildx install
docker buildx create --name arm64-builder --platform linux/arm64
docker buildx use arm64-builder
docker pull multiarch/qemu-user-static
docker run --rm --privileged multiarch/qemu-user-static:register --reset
```
```bash
docker build . --network host --platform linux/arm64 --pull --load -t {IMAGE_NAME} --build-arg ML_MODEL_ID_arg=ML_MODEL_ID --build-arg PROJECT_VERSION_arg=PROJECT_VERSION
docker save -o {PATH}/oor-docker-image.tar {IMAGE_NAME}
```
Remember to replace with the correct values: ML_MODEL_ID, PROJECT_VERSION.


On the device:
The device needs to have NVIDIA Container Toolkit installed and docker configured to run with gpus.
NVIDIA Container Toolkit:
```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
  && curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
    sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
    sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit
```
Docker configuration:
```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```
```bash
sudo docker load -i {TAR_IMAGE_PATH}
sudo docker run -d --restart unless-stopped --mount type=bind,source={logs_path},target=/cvt_logs --mount type=bind,source={detections_path},target=/detections --mount type=bind,source={temp_metadata_path},target=/temp_metadata --mount type=bind,source={training_mode_output_path},target=/training_mode --mount type=bind,source={model_path},target=/model_artifacts --mount type=bind,source={input_path},target=/raw_frames -e SHARED_ACCESS_KEY_IOT={SHARED_ACCESS_KEY_IOT} -e AI_INSTRUMENTATION_KEY={AI_INSTRUMENTATION_KEY}  --gpus all --runtime nvidia acroorontweuitr01.azurecr.io/oor-model-arm64-v8
```
Remember to replace with the correct values: SHARED_ACCESS_KEY_IOT, AI_INSTRUMENTATION_KEY.
