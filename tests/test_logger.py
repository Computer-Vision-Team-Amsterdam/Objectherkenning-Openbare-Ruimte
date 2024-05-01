import logging
from aml_interface.azure_logging import AzureLoggingConfigurer


def example_logger_functionality():

    # I'm adding this as a dict directly since I saw you put the rest of the functionality in the other PR
    config = {
            "loglevel_own": "INFO",
            "own_packages": ["__main__", "objectherkenning_openbare_ruimte"],
            "basic_config": {
                "level": "INFO",
                "format": "%(asctime)s|||%(levelname)-8s|%(name)s|%(message)s",
                "datefmt": "%Y-%m-%d %H:%M:%S"
            },
            "ai_instrumentation_key": "INSERT_KEY_HERE"
        }

    azureLoggingConfigurer = AzureLoggingConfigurer(config, __name__)
    azureLoggingConfigurer.setup_oor_logging()
    logger = logging.getLogger("objectherkenning_openbare_ruimte")
    logger.info("This is a test log.")


example_logger_functionality()
