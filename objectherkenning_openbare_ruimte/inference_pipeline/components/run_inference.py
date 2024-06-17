import logging
import os
import sys

sys.path.append("../../..")

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

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
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("inference_pipeline")

from azure.ai.ml.constants import AssetTypes  # noqa: E402
from mldesigner import Input, Output, command_component  # noqa: E402

from objectherkenning_openbare_ruimte.inference_pipeline.source.data_inference import (  # noqa: E402
    DataInference,
)


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
    # image_paths = find_image_paths(mounted_dataset)
    model_name = settings["inference_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    detection_params = settings["inference_pipeline"]["detection_params"]
    batch_size = settings["inference_pipeline"]["detection_params"]["batch_size"]
    prelabeling_flag = settings["inference_pipeline"]["prelabeling_flag"]

    params = {
        "imgsz": detection_params.get("img_size", 640),
        "save": detection_params.get("save_img_flag", False),
        "save_txt": detection_params.get("save_txt_flag", False),
        "save_conf": detection_params.get("save_conf_flag", False),
        "conf": detection_params.get("conf", 0.25),
        "project": project_path,
    }

    inference_pipeline = DataInference(
        images_folder=mounted_dataset,
        inference_folder=project_path,
        model_name=model_name,
        pretrained_model_path=pretrained_model_path,
        inference_params=params,
        target_classes=[2, 3, 4],
        sensitive_classes=[0, 1],
        batch_size=batch_size,
    )

    if prelabeling_flag:
        inference_pipeline.run_pipeline_prelabeling()
    else:
        inference_pipeline.run_pipeline()
