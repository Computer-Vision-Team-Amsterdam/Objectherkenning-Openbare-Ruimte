from cvtoolkit.helpers.file_helpers import find_image_paths


class DataDelivery:
    def __init__(self, images_path: str, detections_path: str, metadata_path: str):
        self.images_path = images_path
        self.detections_path = detections_path
        self.metadata_path = metadata_path

    def run_pipeline(self):
        print(f"Running data delivery pipeline on {self.images_path}..")
        self._match_metadata_to_images()
        self._deliver_data()
        self._delete_data()

    def _match_metadata_to_images(self):
        print(f"Collecting data from {self.images_path}..")
        image_paths = find_image_paths(self.images_path)
        print(f"Data: {image_paths}..")

    def _deliver_data(self):
        print(f"Delivering data from {self.images_path}..")

    def _delete_data(self):
        print(f"Deleting data from {self.images_path}..")
