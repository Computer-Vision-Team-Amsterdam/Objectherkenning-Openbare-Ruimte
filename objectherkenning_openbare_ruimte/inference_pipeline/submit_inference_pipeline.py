import os

from aml_interface.aml_interface import AMLInterface
from azure.ai.ml import Input, Output
from azure.ai.ml.constants import AssetTypes
from azure.ai.ml.dsl import pipeline

from objectherkenning_openbare_ruimte.inference_pipeline.components.run_inference import (
    run_inference,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

aml_interface = AMLInterface()


@pipeline()
def inference_pipeline():

    input_datastore_path = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["inputs"]["datastore_path"]
    )
    output_datastore_path = aml_interface.get_datastore_full_path(
        settings["inference_pipeline"]["outputs"]["datastore_path"]
    )

    inference_data_rel_path = settings["inference_pipeline"]["inputs"][
        "inference_data_rel_path"
    ]
    model_weights_rel_path = settings["inference_pipeline"]["inputs"][
        "model_weights_rel_path"
    ]
    output_rel_path = settings["inference_pipeline"]["outputs"]["output_rel_path"]

    inference_data_path = os.path.join(input_datastore_path, inference_data_rel_path)
    inference_data = Input(
        type=AssetTypes.URI_FOLDER,
        path=inference_data_path,
    )

    model_weights_path = os.path.join(input_datastore_path, model_weights_rel_path)
    model_weights = Input(
        type=AssetTypes.URI_FOLDER,
        path=model_weights_path,
    )
    run_inference_step = run_inference(
        inference_data_dir=inference_data, model_weights_dir=model_weights
    )

    output_path = os.path.join(output_datastore_path, output_rel_path)
    run_inference_step.outputs.output_dir = Output(
        type="uri_folder", mode="rw_mount", path=output_path
    )

    return {}


def main() -> None:
    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface.submit_pipeline_experiment(
        inference_pipeline, "inference_pipeline", default_compute
    )


if __name__ == "__main__":
    main()
