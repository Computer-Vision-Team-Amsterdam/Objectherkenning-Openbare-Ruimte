from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from objectherkenning_openbare_ruimte.convert_annotations_pipeline.components.convert_annotations import (
    convert_annotations,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def convert_annotations_pipeline():

    input_datastore_name = settings["convert_annotations"]["input_datastore_name"]
    input_old_path = aml_interface.get_datastore_full_path(input_datastore_name)

    input_old_input = Input(
        type=AssetTypes.URI_FOLDER,
        path=input_old_path,
        description="Path to the folder containing the annotations to convert",
    )

    final_datastore_name = settings["convert_annotations"]["final_datastore_name"]
    categories_file = settings["convert_annotations"]["categories_file"]
    separate_labels = settings["convert_annotations"]["separate_labels"]
    label_folder = settings["convert_annotations"]["label_folder"]

    convert_annotations_step = convert_annotations(
        input_old_folder=input_old_input,
        datastore_name=final_datastore_name,
        categories_file=categories_file,
        separate_labels=separate_labels,
        label_folder=label_folder,
    )

    output_datastore_name = settings["convert_annotations"]["output_datastore_name"]
    output_new_path = aml_interface.get_datastore_full_path(output_datastore_name)

    convert_annotations_step.outputs.output_new_folder = Output(
        type=AssetTypes.URI_FOLDER,
        path=output_new_path,
        description="Path to the folder containing the converted annotations",
    )

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        convert_annotations_pipeline, "convert_annotations", default_compute
    )
