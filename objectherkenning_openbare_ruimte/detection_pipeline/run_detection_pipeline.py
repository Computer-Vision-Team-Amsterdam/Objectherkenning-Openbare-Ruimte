import logging
import time
from datetime import datetime

from objectherkenning_openbare_ruimte.detection_pipeline.components.data_detection import (
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
    detection_pipeline = DataDetection(
        images_folder=settings["detection_pipeline"]["images_path"],
        detections_folder=settings["detection_pipeline"]["detections_path"],
        model_name=settings["detection_pipeline"]["model_name"],
        pretrained_model_path=settings["detection_pipeline"]["pretrained_model_path"],
        inference_params=settings["detection_pipeline"]["inference_params"],
    )
    while True:
        try:
            logger.info("Running the detection pipeline..")
            detection_pipeline.run_pipeline()
        except Exception as e:
            logger.info(f"Exception occurred in container detection: {e}")
        time.sleep(30)
