import logging
import os
import sys

import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from ultralytics import YOLO

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
log_settings = settings["logging"]
setup_azure_logging(log_settings, __name__)
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("train_model")


@command_component(
    name="train_model",
    display_name="Train a YOLOv8 model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def train_model(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    project_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to train a YOLOv8 model.

    Parameters
    ----------
    mounted_dataset:
        Dataset to use for training, it should contain the following folder structure:
            - /images/train/
            - /images/val/
            - /images/test/
    model_weights:
        Path to the pretrained model weights.
    yolo_yaml_path:
        Location where to store the yaml file for yolo training.
    project_path:
        Location where to store the outputs of the model.
    """
    data = dict(
        path=f"{mounted_dataset}",
        train="images/train/",
        val="images/val/",
        test="images/test/",
        nc=5,
        names=["person", "licence plate", "container", "mobile toilet", "scaffolding"],
    )
    yaml_path = os.path.join(yolo_yaml_path, "yolov8_cfg_nc_5.yaml")
    with open(f"{yaml_path}", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    model_name = settings["training_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    model_parameters = settings["training_pipeline"]["model_parameters"]

    logger.info(f"Pretrained_model_path: {pretrained_model_path}")
    logger.info(f"Model_parameters: {model_parameters}")
    logger.info(f"Project_path: {project_path}")
    logger.info(f"yaml_path: {yaml_path}")
    logger.info(f"Data: {data}")

    model = YOLO(model=pretrained_model_path, task="detect")

    # Prepare dynamic parameters for training
    train_params = {
        "data": yaml_path,
        "epochs": model_parameters.get("epochs", 10),
        "imgsz": model_parameters.get("img_size", 640),
        "project": project_path,
    }

    if "batch" in model_parameters:
        train_params["batch"] = model_parameters["batch"]

    if "patience" in model_parameters:
        train_params["patience"] = model_parameters["patience"]

    # Train the model
    model.train(**train_params)
