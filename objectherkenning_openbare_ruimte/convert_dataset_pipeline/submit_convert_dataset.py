from aml_interface.azure_logging import setup_azure_logging  # noqa: E402
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from objectherkenning_openbare_ruimte.convert_dataset_pipeline.components.convert_dataset import (
    convert_dataset,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
setup_azure_logging(settings["logging"], __name__)

from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def convert_dataset_pipeline():

    input_old_path = aml_interface.get_datastore_full_path("dataset_conversion_old")

    input_old_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_old_path,
        description="Path to the folder containing the dataset to convert",
    )

    face_width = settings["convert_dataset"]["face_width"]

    convert_dataset_step = convert_dataset(
        input_old_folder=input_old_input,
        face_width=face_width,
    )

    output_new_path = aml_interface.get_datastore_full_path("dataset_conversion_new")

    convert_dataset_step.outputs.output_new_folder = Output(
        type=AssetTypes.URI_FOLDER,
        path=output_new_path,
        description="Path to the folder containing the converted dataset",
    )

    print(input_old_input)
    print(output_new_path)
    print(face_width)

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        convert_dataset_pipeline, "convert_dataset", default_compute
    )
