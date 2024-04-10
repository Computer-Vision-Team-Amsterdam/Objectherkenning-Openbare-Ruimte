import logging
import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from ultralytics import YOLO

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402
from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

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
logger = logging.getLogger("run_inference")


@command_component(
    name="run_inference",
    display_name="Run inference on YOLOv8 model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def run_inference(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    project_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to run inference on YOLOv8 model.
    """
    image_paths = find_image_paths(mounted_dataset)
    model_name = settings["inference_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    inference_params = settings["inference_pipeline"]["inference_params"]

    logger.info(f"Image_paths: {image_paths}")
    logger.info(f"Inference_params: {inference_params}")
    logger.info(f"Pretrained_model_path: {pretrained_model_path}")
    logger.info(f"Yolo model: {model_name}")
    logger.info(f"Project_path: {project_path}")

    model = YOLO(model=pretrained_model_path, task="detect")

    # Prepare dynamic parameters for inference
    inference_params = {
        "source": image_paths,
        "imgsz": inference_params.get("img_size", 640),
        "save": inference_params.get("save_img_flag", False),
        "save_txt": inference_params.get("save_txt_flag", False),
        "conf": inference_params.get("conf", 0.25),
    }

    # Train the model
    results = model(**inference_params)
    print(results)
