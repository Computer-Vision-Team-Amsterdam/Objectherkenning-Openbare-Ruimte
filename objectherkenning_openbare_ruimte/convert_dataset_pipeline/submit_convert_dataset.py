from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

# from azure.ai.ml import Input, Output
# from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

# from objectherkenning_openbare_ruimte.convert_dataset_pipeline.components.convert_dataset import (
#    convert_dataset,
# )
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
setup_azure_logging(settings["logging"], __name__)

from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def convert_dataset_pipeline():
    input_path = aml_interface.get_datastore_full_path("old_dataset")

    print(input_path)


if __name__ == "__main__":
    # Retrieve values from the YAML
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        convert_dataset_pipeline, "convert_dataset", default_compute
    )
