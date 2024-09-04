import os
from datetime import datetime
from typing import Any, Dict

from aml_interface.azure_logging import AzureLoggingConfigurer


def setup_luna_logging(logging_cfg: Dict[str, Any], logging_file_path: str):
    """
    Sets up logging according to the configuration.

    Parameters
    ----------
    logging_cfg: logging configuration, for example:
        logging:
          loglevel_own: INFO  # override loglevel for packages defined in `own_packages`
          own_packages: ["__main__", "custom_package_1", "custom_package_2"]
          basic_config:
            # log config as arguments to `logging.basicConfig`
            level: INFO
            format: "%(asctime)s|||%(levelname)-8s|%(name)s|%(message)s"
            datefmt: "%Y-%m-%d %H:%M:%S"
          luna_logs_dir: "/cvt_logs"
    logging_file_path: path where to store the log file
    """
    logging_cfg["basic_config"]["filemode"] = "a"
    logging_cfg["basic_config"]["filename"] = logging_file_path
    os.makedirs(os.path.dirname(logging_file_path), exist_ok=True)
    with open(logging_file_path, "w") as f:
        f.write(f"Starting logging: {datetime.now()}")

    azure_logging_configurer = AzureLoggingConfigurer(logging_cfg, __name__)
    azure_logging_configurer.setup_oor_logging()
