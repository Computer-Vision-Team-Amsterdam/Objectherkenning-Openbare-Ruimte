import csv
import os

from cvtoolkit.helpers.file_helpers import delete_folder, find_image_paths

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)


class DataDelivery:
    def __init__(
        self, images_folder: str, detections_folder: str, metadata_folder: str
    ):
        self.images_folder = images_folder
        self.detections_folder = detections_folder
        self.metadata_folder = metadata_folder

    def run_pipeline(self):
        print(f"Running data delivery pipeline on {self.images_folder}..")
        images_and_frames = self._match_metadata_to_images()
        self._deliver_data(images_and_frames=images_and_frames)
        self._delete_data(images_and_frames=images_and_frames)

    def _match_metadata_to_images(self):
        images_paths = find_image_paths(root_folder=self.images_folder)
        images_and_frames = self._get_images_and_frame_numbers(
            images_paths=images_paths
        )
        self._create_filtered_metadata_files(images_and_frames=images_and_frames)
        return images_and_frames

    def _create_filtered_metadata_files(self, images_and_frames):
        for image_name, frame_numbers in images_and_frames.items():
            filtered_rows = []
            with open(f"{self.metadata_folder}/{image_name}.csv") as fd:
                reader = csv.reader(fd)
                header = next(reader)
                header.append("frame_number")
                filtered_rows.append(header)
                for idx, row in enumerate(reader):
                    if idx + 1 in frame_numbers:
                        row.append(idx + 1)
                        filtered_rows.append(row)
            with open(
                f"{self.images_folder}/{image_name}/{image_name}.csv", "w", newline=""
            ) as output_file:
                csv_writer = csv.writer(output_file)
                csv_writer.writerows(filtered_rows)

    @staticmethod
    def _get_images_and_frame_numbers(images_paths):
        images_and_frames = {}
        for path in images_paths:
            video_name, frame_info = os.path.basename(path).rsplit("_frame_", 1)
            frame_number, _ = frame_info.rsplit(".", 1)
            frame_number = int(frame_number)
            if video_name not in images_and_frames:
                images_and_frames[video_name] = [frame_number]
            else:
                images_and_frames[video_name].append(frame_number)
        return images_and_frames

    def _deliver_data(self, images_and_frames):
        iot_handler = IoTHandler()
        for image_name, _ in images_and_frames.items():
            image_folder = f"{self.images_folder}/{image_name}/"
            print(f"Delivering data from {image_folder}..")
            files = [
                f
                for f in os.listdir(image_folder)
                if os.path.isfile(os.path.join(image_folder, f))
            ]
            for file in files:
                iot_handler.upload_file(os.path.join(image_folder, file))

    def _delete_data(self, images_and_frames):
        for image_name, _ in images_and_frames.items():
            image_folder = f"{self.images_folder}/{image_name}/"
            delete_folder(image_folder)
