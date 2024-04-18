import csv
import os
from typing import Dict, List

from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)


class DataDelivery:
    def __init__(
        self, images_folder: str, detections_folder: str, metadata_folder: str
    ):
        """

        Parameters
        ----------
        images_folder
            Folder containing the blurred images with containers detected
        detections_folder
            Folder containing txt files with detections per image
        metadata_folder
            Folder containing the metadata files in csv format
        """
        self.images_folder = images_folder
        self.detections_folder = detections_folder
        self.metadata_folder = metadata_folder
        self.iot_settings = ObjectherkenningOpenbareRuimteSettings.get_settings()[
            "azure_iot"
        ]

    def run_pipeline(self):
        """
        Runs the data delivery pipeline:
            - matches metadata to images;
            - delivers the data to Azure;
            - deletes the delivered data.
        """
        print(f"Running data delivery pipeline on {self.images_folder}..")
        images_and_frames = self._match_metadata_to_images()
        print(f"Images and frames: {images_and_frames}")
        self._deliver_data(images_and_frames=images_and_frames)
        self._delete_data(images_and_frames=images_and_frames)

    def _match_metadata_to_images(self) -> Dict[str, List[int]]:
        """
        Creates a csv file containing only the metadata of images with containers.
        Returns video names and which frames contains containers.

        Returns
        -------
        Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        images_paths = find_image_paths(root_folder=self.images_folder)
        videos_and_frames = self._get_videos_and_frame_numbers(
            images_paths=images_paths
        )
        self._create_filtered_metadata_files(videos_and_frames=videos_and_frames)
        return videos_and_frames

    def _create_filtered_metadata_files(
        self, videos_and_frames: Dict[str, List[int]]
    ) -> None:
        """
        Creates a metadata file containing only metadata information of frames with containers.

        Parameters
        ----------
        videos_and_frames
            Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        for video_name, frame_numbers in videos_and_frames.items():
            filtered_rows = []
            file_path_only_filtered_rows = (
                f"{self.images_folder}/{video_name}/{video_name}.csv"
            )
            already_existing_frames = []
            with open(f"{self.metadata_folder}/{video_name}.csv") as metadata_file:
                reader = csv.reader(metadata_file)
                header = next(reader)
                header.append("frame_number")
                if not os.path.isfile(file_path_only_filtered_rows):
                    filtered_rows.append(header)
                else:
                    already_existing_frames = self._get_frame_numbers_in_metadata_file(
                        file_path_only_filtered_rows
                    )
                for idx, row in enumerate(reader):
                    if (
                        idx + 1 in frame_numbers
                        and idx + 1 not in already_existing_frames
                    ):
                        row.append(str(idx + 1))
                        filtered_rows.append(row)
            if filtered_rows:
                with open(file_path_only_filtered_rows, "a", newline="") as output_file:
                    csv_writer = csv.writer(output_file)
                    csv_writer.writerows(filtered_rows)

    @staticmethod
    def _get_frame_numbers_in_metadata_file(file_path):
        frame_numbers = []
        with open(file_path) as metadata_file:
            reader = csv.reader(metadata_file)
            for row in reader:
                if row:
                    try:
                        last_column_value = int(row[-1])
                        frame_numbers.append(last_column_value)
                    except ValueError:
                        pass
        return frame_numbers

    @staticmethod
    def _get_videos_and_frame_numbers(images_paths: List[str]) -> Dict[str, List[int]]:
        """
        Starting from a list of paths, groups all the frame numbers under the same video name.

        Parameters
        ----------
        images_paths
            List of frames path. The frame name are structured: video_name _frame_ frame_number. Ex. video1_frame_1

        Returns
        -------
        Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        videos_and_frames = {}
        for path in images_paths:
            video_name, frame_info = os.path.basename(path).rsplit("_frame_", 1)
            frame_number, _ = frame_info.rsplit(".", 1)
            if video_name not in videos_and_frames:
                videos_and_frames[video_name] = [int(frame_number)]
            else:
                videos_and_frames[video_name].append(int(frame_number))
        return videos_and_frames

    def _deliver_data(self, images_and_frames: Dict[str, List[str]]):
        """
        Using Azure IoT delivers the images and metadata to Azure.

        Parameters
        ----------
        images_and_frames
            Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        iot_handler = IoTHandler(
            hostname=self.iot_settings["hostname"],
            device_id=self.iot_settings["device_id"],
            # shared_access_key=self.iot_settings["shared_access_key"],
            cert_file_path=self.iot_settings["cert_file_path"],
            key_file_path=self.iot_settings["key_file_path"],
            passphrase=self.iot_settings["passphrase"],
        )
        for image_name, frame_numbers in images_and_frames.items():
            image_folder = f"{self.images_folder}/{image_name}"
            iot_handler.upload_file(f"{image_folder}/{image_name}.csv")
            for frame_number in frame_numbers:
                iot_handler.upload_file(
                    f"{image_folder}/{image_name}_frame_{frame_number}.jpg"
                )

    def _delete_data(self, images_and_frames: Dict[str, List[str]]):
        """
        Deletes the data that has been delivered to Azure.

        Parameters
        ----------
        images_and_frames
            Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        for image_name, frame_numbers in images_and_frames.items():
            image_folder = f"{self.images_folder}/{image_name}"
            delete_file(f"{image_folder}/{image_name}.csv")
            for frame_number in frame_numbers:
                delete_file(f"{image_folder}/{image_name}_frame_{frame_number}.jpg")
