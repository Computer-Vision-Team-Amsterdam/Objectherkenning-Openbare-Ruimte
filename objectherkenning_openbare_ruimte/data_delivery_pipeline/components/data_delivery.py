import json

from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)


class DataDelivery:
    def __init__(self, images_path: str, detections_path: str, metadata_path: str):
        self.images_path = images_path
        self.detections_path = detections_path
        self.metadata_path = metadata_path

    def run_pipeline(self):
        print(f"Running data delivery pipeline on {self.images_path}..")
        # self._match_metadata_to_images()
        self._deliver_data()
        # self._delete_data()

    def _match_metadata_to_images(self):
        print(f"Collecting data from {self.images_path}..")
        image_paths = find_image_paths(self.images_path)
        print(f"Data: {len(image_paths)}..")

    def _deliver_data(self):
        print(f"Delivering data from {self.images_path}..")
        iot_handler = IoTHandler()
        message = {"test": "Test message from Sebastian."}
        iot_handler.deliver_message(json.dumps(message))
        # iot_handler.upload_file("Picture1.jpg")

    def _delete_data(self):
        print(f"Deleting data from {self.images_path}..")
        map(delete_file, find_image_paths(self.images_path))
