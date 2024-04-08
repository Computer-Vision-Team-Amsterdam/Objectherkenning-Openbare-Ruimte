import logging
import os
import sys

import torch
from mldesigner import command_component

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
log_settings = settings["logging"]
setup_azure_logging(log_settings, __name__)
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("cuda_test")


@command_component(
    name="cuda_test",
    display_name="Cuda test.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def cuda_test():
    if torch.cuda.is_available():
        print(f"CUDA is available. Number of GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Please check your installation.")
