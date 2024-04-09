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
        settings["frame_extraction"]["inputs"]["datastore"]
    )
    input_rel_path = settings["frame_extraction"]["inputs"]["rel_path"]
    input_data_path = os.path.join(input_datastore_path, input_rel_path)

    output_datastore_path = aml_interface.get_datastore_full_path(
        settings["frame_extraction"]["outputs"]["datastore"]
    )
    output_rel_path = settings["frame_extraction"]["outputs"]["rel_path"]
    output_data_path = os.path.join(output_datastore_path, output_rel_path)

    job = command(
        code=".",
        command='poetry run python objectherkenning_openbare_ruimte/frame_extraction/frame_extraction.py --input_folder "${{inputs.input_folder}}" --output_folder "${{outputs.output_folder}}"',
        inputs={
            "input_folder": Input(
                path=input_data_path,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RO_MOUNT,
            )
        },
        outputs={
            "output_folder": Output(
                path=output_data_path,
                type=AssetTypes.URI_FOLDER,
                mode=InputOutputModes.RW_MOUNT,
            )
        },
        compute=compute,
        environment=f"{env_name}:{env_version}",
    )

    # submit job
    submitted_job = aml_interface.submit_command_job(job)
    aml_url = submitted_job.studio_url
    print("Monitor your job at", aml_url)
