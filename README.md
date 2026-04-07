# Objectherkenning-Openbare-Ruimte

This project is about recognizing objects in public space (street view) images. It includes generic code for inference and model evaluation, and a databricks pipeline for post-processing of the results.

In our current solution the databricks pipeline is used together with IoT edge processing using the code in the [OOR-on-edge](https://github.com/Computer-Vision-Team-Amsterdam/OOR-on-edge) repo.


## Installation

### 1. Clone the code

```bash
git clone git@github.com:Computer-Vision-Team-Amsterdam/Objectherkenning-Openbare-Ruimte.git
```

### 2. Install UV
We use UV as package manager, which can be installed using any method mentioned on [the UV webpage](https://docs.astral.sh/uv/getting-started/installation/).

The easiest option is to use their installer:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

It is also possible to use pip:
```bash
pipx install uv
```

Afterwards, uv can be updated using `uv self update`.

### 3. Install dependencies
In the terminal, navigate to the project root (the folder containing `pyproject.toml`), then use UV to create a new virtual environment and install the dependencies.

```bash
# Create the environment locally in the folder .venv
uv venv --python 3.11

# Activate the environment
source .venv/bin/activate 

# Install dependencies
uv pip install -r pyproject.toml --extra dev

# Add package
uv add <package_name>
```

### 4. Install pre-commit hooks
The pre-commit hooks help to ensure that all committed code is valid and consistently formatted. We use UV to manage pre-commit as well.

```bash
uv tool install pre-commit --with pre-commit-uv --force-reinstall

# Install pre-commit hooks
pre-commit install

# Optional: update pre-commit hooks
pre-commit autoupdate

# Run pre-commit hooks using
bash .git/hooks/pre-commit
```
