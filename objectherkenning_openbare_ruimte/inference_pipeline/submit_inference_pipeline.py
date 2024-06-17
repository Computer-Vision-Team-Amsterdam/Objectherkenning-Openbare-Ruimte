import os

from aml_interface.azure_logging import AzureLoggingConfigurer  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
azure_logging_configurer = AzureLoggingConfigurer(settings["logging"])
azure_logging_configurer.setup_oor_logging()

from aml_interface.aml_interface import AMLInterface  # noqa: E402
from azure.ai.ml import Input, Output  # noqa: E402
from azure.ai.ml.constants import AssetTypes  # noqa: E402
from azure.ai.ml.dsl import pipeline  # noqa: E402

from objectherkenning_openbare_ruimte.inference_pipeline.components.run_inference import (  # noqa: E402
    run_inference,
)


@pipeline()
def inference_pipeline():

    datastore_path = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["inputs"]["datastore_path"]
    )
    inference_data_rel_path = settings["inference_pipeline"]["inputs"][
        "inference_data_rel_path"
    ]
    model_weights_rel_path = settings["inference_pipeline"]["inputs"][
        "model_weights_rel_path"
    ]
    project_rel_path = settings["inference_pipeline"]["outputs"]["project_rel_path"]

    inference_data_path = os.path.join(datastore_path, inference_data_rel_path)
    inference_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=inference_data_path,
    )

    model_weights_path = os.path.join(datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )
    run_inference_step = run_inference(
        mounted_dataset=inference_data, model_weights=model_weights
    )

    project_path = os.path.join(datastore_path, project_rel_path)
    run_inference_step.outputs.project_path = Output(
        type="uri_folder", mode="rw_mount", path=project_path
    )

    return {}


if __name__ == "__main__":
    # Retrieve values from the YAML
    ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        inference_pipeline, "inference_pipeline", default_compute
    )
