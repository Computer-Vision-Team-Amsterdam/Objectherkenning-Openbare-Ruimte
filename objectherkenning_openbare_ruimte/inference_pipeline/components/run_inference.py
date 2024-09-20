import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.inference_pipeline.source.data_inference import (  # noqa: E402
    DataInference,
)
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

azure_logging_configurer = AzureLoggingConfigurer(settings["logging"])
azure_logging_configurer.setup_oor_logging()
logger = logging.getLogger("inference_pipeline")

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="inference_pipeline",
    display_name="Run inference using YOLOv8 model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def run_inference(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to run inference on YOLOv8 model.
    """
    inference_setting = settings["inference_pipeline"]
    model_name = inference_setting["inputs"]["model_name"]
    model_path = os.path.join(model_weights, model_name)
    model_params = inference_setting["model_params"]
    batch_size = model_params["batch_size"]

    full_model_params = {
        "imgsz": model_params.get("img_size", 640),
        "save": model_params.get("save_img_flag", False),
        "save_txt": model_params.get("save_txt_flag", False),
        "save_conf": model_params.get("save_conf_flag", False),
        "conf": model_params.get("conf", 0.25),
        "project": output_path,
    }

    inference_pipeline = DataInference(
        images_folder=mounted_dataset,
        output_folder=output_path,
        model_path=model_path,
        inference_params=full_model_params,
        target_classes=inference_setting["target_classes"],
        sensitive_classes=inference_setting["sensitive_classes"],
        target_classes_conf=inference_setting["target_classes_conf"],
        sensitive_classes_conf=inference_setting["sensitive_classes_conf"],
        output_image_size=inference_setting["output_image_size"],
        save_detections=inference_setting["save_detection_images"],
        save_labels=inference_setting["save_detection_labels"],
        detections_subfolder=inference_setting["outputs"]["detections_subfolder"],
        labels_subfolder=inference_setting["outputs"]["labels_subfolder"],
        defisheye_flag=inference_setting["defisheye_flag"],
        defisheye_params=settings["distortion_correction"]["defisheye_params"],
        batch_size=batch_size,
    )

    inference_pipeline.run_pipeline()
