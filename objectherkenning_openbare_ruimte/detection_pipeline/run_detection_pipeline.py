import time

from objectherkenning_openbare_ruimte.detection_pipeline.components.data_detection import (
    DataDetection,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    while True:
        try:
            detection_pipeline = DataDetection(
                images_folder=settings["data_delivery_pipeline"]["images_path"],
                detections_folder=settings["data_delivery_pipeline"]["detections_path"],
            )
            detection_pipeline.run_pipeline()
        except Exception as e:
            print(f"Exception occurred in container detection: {e}")
        time.sleep(30)
