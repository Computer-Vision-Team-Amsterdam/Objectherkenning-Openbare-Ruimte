import logging
import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
log_settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)[
    "logging"
]
setup_azure_logging(log_settings, __name__)
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("convert_annotations")


@command_component(
    name="convert_annotations",
    display_name="Converts annotations from YOLO to Azure COCO format.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def convert_annotations(
    input_old_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_new_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to convert annotations from YOLO to Azure COCO format.

    Parameters
    ----------
    input_old_folder:
        Path to the folder containing the annotations to convert.
    output_new_folder:
        Path to the folder containing the converted annotations.
    """

    logger.info(f"Input folder: {input_old_folder}")
