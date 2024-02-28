import logging
import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402
from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

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
logger = logging.getLogger("convert_old_dataset")

from objectherkenning_openbare_ruimte.convert_dataset_pipeline.source.equirectangular_to_cubemap_converter import (  # noqa: E402
    EquirectangularToCubemapConverter,
)


@command_component(
    name="convert_dataset",
    display_name="Converts a dataset with equirectangular images to a dataset with cubemaps.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def convert_dataset(
    input_old_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_new_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    face_width: int,
):
    """
    Pipeline step to convert a dataset with equirectangular images to
    a dataset with cubemaps. This includes images and annotations.

    Parameters
    ----------
    input_old_folder:
        Path to the folder containing the dataset to convert.
    output_new_folder:
        Path to the folder containing the converted dataset.
    face_width: int
        Width of each face in the cubemap.
    """

    image_paths = find_image_paths(input_old_folder)
    logger.info(f"Input folder: {input_old_folder}")
    logger.info(f"Output folder: {output_new_folder}")
    logger.info(f"Face width: {face_width}")
    logger.info(f"Number of images: {len(image_paths)}")

    equirectangular_to_cubemap_converter = EquirectangularToCubemapConverter(
        input_old_folder, output_new_folder, face_width
    )

    for img_path in image_paths:
        equirectangular_to_cubemap_converter.convert_image(img_path)
        equirectangular_to_cubemap_converter.process_annotations(img_path)
        logging.info(f"Processed {img_path}")
