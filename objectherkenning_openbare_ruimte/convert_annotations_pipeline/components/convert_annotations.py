import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.convert_annotations_pipeline.source.yolo_to_azure_coco_converter import (  # noqa: E402
    YoloToAzureCocoConverter,
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
logger = logging.getLogger("convert_annotations_pipeline")

aml_experiment_settings = settings["aml_experiment_details"]


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
    datastore_name: str,
    image_storage_account: str,
    categories_file: str,
    separate_labels: bool = False,
    label_folder: str = None,
):
    """
    Pipeline step to convert annotations from YOLO to Azure COCO format.

    Parameters
    ----------
    input_old_folder:
        Path to the folder containing the annotations to convert.
    output_new_folder:
        Path to the folder containing the converted annotations.
    datastore_name: str
        Name of the datastore of the dataset.
    image_storage_account: str
        Name of the storage account of the dataset.
    categories_file: str
        Path to the JSON file containing the categories.
    separate_labels: bool, optional
        Whether the labels are stored in a separate folder.
    label_folder: str, optional
        Path to the folder containing the label files if separate.
    """

    # image_paths = find_image_paths(input_old_folder)
    logger.info(f"Input folder: {input_old_folder}")
    logger.info(f"Output folder: {output_new_folder}")
    logger.info(f"Datastore name: {datastore_name}")
    logger.info(f"Image storage account: {image_storage_account}")
    logger.info(f"Categories file: {categories_file}")
    logger.info(f"Separate labels: {separate_labels}")

    categories_file = os.path.join(input_old_folder, categories_file)
    label_folder_path = None

    if separate_labels:
        label_folder_path = os.path.join(input_old_folder, label_folder)
        logger.info(f"Label folder path: {label_folder_path}")

    converter = YoloToAzureCocoConverter(
        input_old_folder,
        output_new_folder,
        datastore_name,
        image_storage_account,
        categories_file,
        separate_labels,
        label_folder_path,
    )
    converter.convert()

    logger.info(f"JSON file created in: {output_new_folder}")
