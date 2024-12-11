import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.inference_pipeline.source.OORInference import (  # noqa: E402
    OORInference,
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
    inference_data_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Run inference using a pretrained YOLOv8 model on a chosen set of images.

    Parameters
    ----------
    inference_data_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of images to run inference on. The optional sub-folder
        structure will be preserved in the output.
    model_weights_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of the model weights.
    output_dir: Output(type=AssetTypes.URI_FOLDER)
        Location where output will be stored. Depending on the config settings
        this can be annotation labels as .txt files, images with blurred
        sensitive classes and bounding boxes, or both.
    """
    inference_settings = settings["inference_pipeline"]

    inference_pipeline = OORInference(
        images_folder=inference_data_dir,
        output_folder=output_dir,
        model_path=os.path.join(
            model_weights_dir, inference_settings["inputs"]["model_name"]
        ),
        inference_settings=inference_settings,
    )

    inference_pipeline.run_pipeline()
