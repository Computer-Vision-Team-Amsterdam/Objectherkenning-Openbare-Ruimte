import logging
import time
import traceback
from datetime import datetime

from objectherkenning_openbare_ruimte.on_edge.detection_pipeline.components.data_detection import (
    DataDetection,
)
from objectherkenning_openbare_ruimte.settings.luna_logging import setup_luna_logging
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    logging_file_path = f"{settings['logging']['luna_logs_dir']}/detection_pipeline/{datetime.now()}.txt"
    setup_luna_logging(settings["logging"], logging_file_path)
    logger = logging.getLogger("detection_pipeline")
    logger.info("Building the detection pipeline..")
    detection_pipeline = DataDetection()
    logger.info(
        f"Running the detection pipeline on {settings['detection_pipeline']['images_path']}.."
    )
    while True:
        try:
            detection_pipeline.run_pipeline()
        except Exception as e:
            logger.error(f"Exception occurred in container detection: {e}")
            logger.error(traceback.format_exc())
        time.sleep(settings["detection_pipeline"]["sleep_time"])
