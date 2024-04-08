from aml_interface.azure_logging import setup_azure_logging  # noqa: E402
from azure.ai.ml.dsl import pipeline

from objectherkenning_openbare_ruimte.cuda_test_pipeline.components.cuda_test import (
    cuda_test,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
setup_azure_logging(settings["logging"], __name__)

from aml_interface.aml_interface import AMLInterface  # noqa: E402


@pipeline()
def cuda_test_pipeline():

    cuda_test_step = cuda_test()

    return cuda_test_step


if __name__ == "__main__":
    # Retrieve values from the YAML
    ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

    default_compute = settings["aml_experiment_details"]["compute_name"]
    aml_interface = AMLInterface()
    aml_interface.submit_pipeline_experiment(
        cuda_test_pipeline, "cuda_test_pipeline", default_compute
    )
