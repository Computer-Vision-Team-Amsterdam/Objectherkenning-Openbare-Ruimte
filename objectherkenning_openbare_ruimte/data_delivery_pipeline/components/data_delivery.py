import csv
import logging
import os
from typing import Dict, List

from cvtoolkit.helpers.file_helpers import delete_file, find_image_paths

from objectherkenning_openbare_ruimte.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

logger = logging.getLogger("data_delivery_pipeline")


class DataDelivery:
    def __init__(self, detections_folder: str, metadata_folder: str):
        """

        Parameters
        ----------
        detections_folder
            Folder containing the blurred images with containers detected
        metadata_folder
            Folder containing the metadata files in csv format
        """
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
        logger.info(f"Running data delivery pipeline on {self.detections_folder}..")
        videos_and_frames = self._match_metadata_to_images()
        logger.info(f"Images and frames: {videos_and_frames}")
        self._deliver_data(videos_and_frames=videos_and_frames)
        self._delete_data(videos_and_frames=videos_and_frames)

    def _match_metadata_to_images(self) -> Dict[str, List[str]]:
        """
        Creates a csv file containing only the metadata of images with containers.
        Returns video names and which frames contains containers.

        Returns
        -------
        Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        images_paths = find_image_paths(root_folder=self.detections_folder)
        videos_and_frames = self._get_videos_and_frame_numbers(
            images_paths=images_paths
        )
        self._create_filtered_metadata_files(videos_and_frames=videos_and_frames)
        return videos_and_frames

    def _create_filtered_metadata_files(
        self, videos_and_frames: Dict[str, List[str]]
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
                f"{self.detections_folder}/{video_name}/{video_name}.csv"
            )
            already_existing_frames = []
            try:
                with open(f"{self.metadata_folder}/{video_name}.csv") as metadata_file:
                    reader = csv.reader(metadata_file)
                    header = next(reader)
                    header.append("frame_number")
                    if not os.path.isfile(file_path_only_filtered_rows):
                        filtered_rows.append(header)
                    else:
                        already_existing_frames = (
                            self._get_frame_numbers_in_metadata_file(
                                file_path_only_filtered_rows
                            )
                        )
                    frame_numbers_int = [int(x) for x in frame_numbers]
                    for idx, row in enumerate(reader):
                        if (
                            idx + 1 in frame_numbers_int
                            and idx + 1 not in already_existing_frames
                        ):
                            row.append(str(idx + 1))
                            filtered_rows.append(row)
                if filtered_rows:
                    with open(
                        file_path_only_filtered_rows, "a", newline=""
                    ) as output_file:
                        csv_writer = csv.writer(output_file)
                        csv_writer.writerows(filtered_rows)
            except FileNotFoundError as e:
                logger.error(
                    f"FileNotFoundError during the creation of metadata file for: {video_name}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Exception during the creation of metadata file for: {video_name}: {e}"
                )

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
    def _get_videos_and_frame_numbers(images_paths: List[str]) -> Dict[str, List[str]]:
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
                videos_and_frames[video_name] = [frame_number]
            else:
                videos_and_frames[video_name].append(frame_number)
        return videos_and_frames

    def _deliver_data(self, videos_and_frames: Dict[str, List[str]]):
        """
        Using Azure IoT delivers the images and metadata to Azure.

        Parameters
        ----------
        videos_and_frames
            Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        iot_handler = IoTHandler(
            hostname=self.iot_settings["hostname"],
            device_id=self.iot_settings["device_id"],
            shared_access_key=self.iot_settings["shared_access_key"],
        )
        for video_name, frame_numbers in videos_and_frames.items():
            detection_folder = f"{self.detections_folder}/{video_name}"
            iot_handler.upload_file(f"{detection_folder}/{video_name}.csv")
            for frame_number in frame_numbers:
                iot_handler.upload_file(
                    f"{detection_folder}/{video_name}_frame_{frame_number}.jpg"
                )

    def _delete_data(self, videos_and_frames: Dict[str, List[str]]):
        """
        Deletes the data that has been delivered to Azure.

        Parameters
        ----------
        videos_and_frames
            Dictionary containing as key a video name and as values the number of frames containing containers.
        """
        for video_name, frame_numbers in videos_and_frames.items():
            detection_folder = f"{self.detections_folder}/{video_name}"
            delete_file(f"{detection_folder}/{video_name}.csv")
            for frame_number in frame_numbers:
                delete_file(f"{detection_folder}/{video_name}_frame_{frame_number}.jpg")
