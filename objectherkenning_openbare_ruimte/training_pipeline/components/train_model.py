import logging
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

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
log_settings = settings["logging"]
setup_azure_logging(log_settings, __name__)
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("convert_old_dataset")


@command_component(
    name="train_model",
    display_name="Train a YOLOv8 model using a defined dataset.",
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
    Pipeline step to train the model.

    Parameters
    ----------
    mounted_dataset:
        Dataset to use for training, it should contain the following folder structure:
            - /images/train/
            - /images/val/
            - /images/test/
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
        nc=3,
        names=["person", "licence plate", "container"],
    )
    with open(f"{yolo_yaml_path}/yolov8_cfg.yaml", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    model_name = settings["training_pipeline"]["model_name"]
    pretrained_model_path = f"{model_weights}/{model_name}"
    model_parameters = settings["training_pipeline"]["model_parameters"]

    model = YOLO(pretrained_model_path)

    # Prepare dynamic parameters for training
    train_params = {
        "data": data,  # This can also be made dynamic if needed
        "epochs": model_parameters.get("epochs", 10),  # Default value if not specified
        "imgsz": model_parameters.get(
            "img_size", 640
        ),  # Default value if not specified
        "project": project_path,
    }

    if "batch_size" in model_parameters:
        train_params["batch_size"] = model_parameters["batch_size"]

    if "patience" in model_parameters:
        train_params["patience"] = model_parameters["patience"]

    # Train the model
    model.train(**train_params)
