import os

from aml_interface.aml_interface import AMLInterface
from azure.ai.ml import Input, Output, command
from azure.ai.ml.constants import AssetTypes, InputOutputModes

from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)
ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


if __name__ == "__main__":
    aml_interface = AMLInterface()

    env_name = settings["aml_experiment_details"]["env_name"]
    env_version = settings["aml_experiment_details"]["env_version"]
    compute = settings["aml_experiment_details"]["compute_name"]

    input_datastore_path = aml_interface.get_datastore_full_path(
        settings["data_minimisation_experiment"]["inputs"]["datastore"]
    )
    input_images_rel_path = settings["data_minimisation_experiment"]["inputs"][
        "images_rel_path"
    ]
    input_images_path = os.path.join(input_datastore_path, input_images_rel_path)
    input_labels_rel_path = settings["data_minimisation_experiment"]["inputs"][
        "labels_rel_path"
    ]
    input_labels_path = os.path.join(input_datastore_path, input_labels_rel_path)

    output_datastore_path = aml_interface.get_datastore_full_path(
        settings["data_minimisation_experiment"]["outputs"]["datastore"]
    )
    output_rel_path = settings["data_minimisation_experiment"]["outputs"]["rel_path"]
    output_data_path = os.path.join(output_datastore_path, output_rel_path)

    image_format = settings["data_minimisation_experiment"]["image_format"]

    cmd = (
        'poetry run python objectherkenning_openbare_ruimte/data_minimisation/data_minimisation.py --images_folder "${{inputs.images_folder}}" --labels_folder "${{inputs.labels_folder}}" --output_folder "${{outputs.output_folder}}" --image_format '
        + image_format
    )

    job = command(
        code=".",
        command=cmd,
        inputs={
            "images_folder": Input(
                path=input_images_path,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
            "labels_folder": Input(
                path=input_labels_path,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            ),
        },
        outputs={
            "output_folder": Output(
                path=output_data_path,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            ),
        },
        compute=compute,
        environment=f"{env_name}:{env_version}",
    )

    # submit job
    submitted_job = aml_interface.submit_command_job(job)
    aml_url = submitted_job.studio_url
    print("Monitor your job at", aml_url)
