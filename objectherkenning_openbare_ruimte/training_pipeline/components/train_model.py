import os
import sys

import wandb
import yaml
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from ultralytics import YOLO
from ultralytics import settings as ultralytics_settings
from wandb.integration.ultralytics import add_wandb_callback

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
aml_experiment_settings = settings["aml_experiment_details"]


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
            - /labels/train/
            - /labels/val/
            - /labels/test/
    model_weights:
        Path to the pretrained model weights.
    yolo_yaml_path:
        Location where to store the yaml file for yolo training.
    project_path:
        Location where to store the outputs of the model.
    """

    ultralytics_settings.update({"runs_dir": project_path})
    wandb.init(job_type="training", config_exclude_keys=["project"])

    n_classes = settings["training_pipeline"]["model_parameters"]["n_classes"]
    name_classes = settings["training_pipeline"]["model_parameters"]["name_classes"]
    data = dict(
        path=f"{mounted_dataset}",
        train="images/train/",
        val="images/val/",
        test="images/test/",
        nc=n_classes,
        names=name_classes,
    )
    yaml_path = os.path.join(yolo_yaml_path, f"oor_dataset_cfg_nc_{n_classes}.yaml")
    with open(f"{yaml_path}", "w") as outfile:
        yaml.dump(data, outfile, default_flow_style=False)

    model_name = settings["training_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    model_parameters = settings["training_pipeline"]["model_parameters"]

    model = YOLO(model=pretrained_model_path, task="detect")

    # Add W&B Callback for Ultralytics
    add_wandb_callback(
        model,
        enable_model_checkpointing=False,
        enable_validation_logging=False,
        enable_prediction_logging=False,
        enable_train_validation_logging=False,
    )

    # Prepare parameters for training
    train_params = {
        "data": yaml_path,
        "epochs": model_parameters.get("epochs", 100),
        "imgsz": model_parameters.get("img_size", 1024),
        "project": project_path,
        "save_dir": project_path,
        "batch": model_parameters.get("batch", -1),
        "patience": model_parameters.get("patience", 100),
        "cos_lr": model_parameters.get("cos_lr", False),
        "dropout": model_parameters.get("dropout", 0.0),
        "seed": model_parameters.get("seed", 0),
        "box": model_parameters.get("box", 7.5),
        "cls": model_parameters.get("cls", 0.5),
        "dfl": model_parameters.get("dfl", 1.5),
    }

    # Train the model
    model.train(**train_params)
