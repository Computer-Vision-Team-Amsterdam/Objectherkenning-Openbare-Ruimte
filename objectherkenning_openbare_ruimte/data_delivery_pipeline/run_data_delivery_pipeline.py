import time
import traceback

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.data_delivery import (
    DataDelivery,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

if __name__ == "__main__":
    settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    data_delivery_pipeline = DataDelivery(
        detections_folder=settings["data_delivery_pipeline"]["detections_path"],
        metadata_folder=settings["data_delivery_pipeline"]["metadata_path"],
    )
    while True:
        try:
            data_delivery_pipeline.run_pipeline()
        except Exception:
            print(f"Exception occurred in data delivery: {traceback.format_exc()}")
        time.sleep(30)
