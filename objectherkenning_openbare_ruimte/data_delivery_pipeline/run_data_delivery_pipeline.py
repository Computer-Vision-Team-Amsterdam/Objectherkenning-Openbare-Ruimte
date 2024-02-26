import time

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.data_delivery import (
    DataDelivery,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    while True:
        try:
            data_delivery_pipeline = DataDelivery(
                images_path="objectherkenning_openbare_ruimte/processed_images",
                detections_path="objectherkenning_openbare_ruimte/processed_images",
                metadata_path="objectherkenning_openbare_ruimte/processed_images",
            )
            data_delivery_pipeline.run_pipeline()
        except Exception as e:
            print(f"Exception occurred in data delivery: {e}")
        time.sleep(5)
