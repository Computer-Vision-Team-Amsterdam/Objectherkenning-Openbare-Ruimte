import json
import os
import sys
from typing import Any, Dict

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
    name="sweep_model",
    display_name="Perform HP sweep with wandb on a YOLOv8 model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def sweep_model(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    yolo_yaml_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    project_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to perform hyperparameter tuning sweep with wandb on a YOLOv8 model.

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

    n_classes = settings["sweep_pipeline"]["model_parameters"]["n_classes"]
    name_classes = settings["sweep_pipeline"]["model_parameters"]["name_classes"]
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

    model_name = settings["sweep_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    model_parameters = settings["sweep_pipeline"]["model_parameters"]
    sweep_trials = settings["sweep_pipeline"]["sweep_trials"]

    train_params = {
        "data": yaml_path,
        "epochs": model_parameters.get("epochs", 100),
        "imgsz": model_parameters.get("img_size", 1024),
        "project": project_path,
        "save_dir": project_path,
        "batch": model_parameters.get("batch", -1),
    }

    def extract_parameter_keys(sweep_config: Dict[str, Any]) -> Any:
        return sweep_config["parameters"].keys()

    def load_sweep_configuration(json_file: str) -> Dict[str, Any]:
        with open(json_file, "r") as file:
            config = json.load(file)
        return config

    # Define the search space
    config_file = settings["sweep_pipeline"]["inputs"]["sweep_config"]
    sweep_configuration = load_sweep_configuration(config_file)

    # Start the sweep
    sweep_id = wandb.sweep(sweep=sweep_configuration)

    def train():
        with wandb.init(job_type="training"):
            config = wandb.config

            # Extract parameter keys from the sweep configuration
            parameter_keys = extract_parameter_keys(sweep_configuration)
            dynamic_params = {
                key: value for key, value in config.items() if key in parameter_keys
            }
            train_params.update(dynamic_params)

            model = YOLO(model=pretrained_model_path, task="detect")

            add_wandb_callback(
                model,
                enable_model_checkpointing=False,
                enable_validation_logging=False,
                enable_prediction_logging=False,
                enable_train_validation_logging=False,
            )
            model.train(**train_params)

    wandb.agent(sweep_id, function=train, count=sweep_trials)
