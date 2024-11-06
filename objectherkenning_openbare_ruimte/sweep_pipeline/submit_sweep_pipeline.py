import os

from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)
from objectherkenning_openbare_ruimte.sweep_pipeline.components.sweep_model import (
    sweep_model,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def sweep_pipeline():

    datastore_path = aml_interface.get_datastore_full_path(
        settings["sweep_pipeline"]["inputs"]["datastore_path"]
    )
    training_rel_path = settings["sweep_pipeline"]["inputs"]["training_data_rel_path"]
    model_weights_rel_path = settings["sweep_pipeline"]["inputs"][
        "model_weights_rel_path"
    ]
    project_datastore_path = aml_interface.get_datastore_full_path(
        settings["sweep_pipeline"]["outputs"]["project_datastore_path"]
    )
    project_rel_path = settings["sweep_pipeline"]["outputs"]["project_rel_path"]

    training_data_path = os.path.join(datastore_path, training_rel_path)
    training_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=training_data_path,
    )

    model_weights_path = os.path.join(datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )
    sweep_model_step = sweep_model(
        mounted_dataset=training_data, model_weights=model_weights
    )
    sweep_model_step.outputs.yolo_yaml_path = Output(
        type="uri_folder", mode="rw_mount", path=model_weights_path
    )

    sweep_model_step.environment_variables = {
        "WANDB_API_KEY": settings["wandb"]["api_key"],
        "WANDB_MODE": settings["wandb"]["mode"],
    }

    project_path = os.path.join(project_datastore_path, project_rel_path)
    sweep_model_step.outputs.project_path = Output(
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
        sweep_pipeline, "sweep_pipeline", default_compute
    )
