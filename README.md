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
docker buildx build --platform linux/arm64 --pull --load -t {IMAGE_NAME} .
docker save -o {PATH}/oor-docker-image.tar {IMAGE_NAME}
```

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
sudo docker load -i {TAR_IMAGE_PAHT}
sudo docker run -d --mount type=bind,source={source_path},target=/raw_videos  --mount type=bind,source={source_path},target=/detections --mount type=bind,source={source_path},target=/raw_frames -e SHARED_ACCESS_KEY_IOT='{shared_access_key_value}' acroorontweuitr01.azurecr.io/oor-model-arm64-v8 
```