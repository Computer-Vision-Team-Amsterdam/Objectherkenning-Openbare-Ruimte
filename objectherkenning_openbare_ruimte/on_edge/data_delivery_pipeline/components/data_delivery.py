import csv
import logging
import os
import pathlib
from datetime import datetime
from typing import List

from cvtoolkit.helpers.file_helpers import delete_file

from objectherkenning_openbare_ruimte.on_edge.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)
from objectherkenning_openbare_ruimte.on_edge.utils import (
    get_frame_metadata_csv_file_paths,
    get_img_name_from_csv_row,
    log_execution_time,
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
            Temporary folder containing the metadata files in csv format before uploading it to Azure
        """
        self.detections_folder = detections_folder
        self.metadata_folder = metadata_folder
        self.settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
        self.metadata_csv_file_paths_with_errors: List[str] = []
        self.model_and_code_version = [
            self.settings["detection_pipeline"]["model_name"],
            self.settings["data_delivery_pipeline"]["ml_model_id"],
            self.settings["data_delivery_pipeline"]["project_version"],
        ]
        self.detection_metadata_header = [
            "image_name",
            "object_class",
            "x_center",
            "y_center",
            "width",
            "height",
            "confidence",
            "tracking_id",
        ]

    @log_execution_time
    def run_pipeline(self):
        """
        Runs the data delivery pipeline:
            - retrieves all the csvs and images that need to be delivered;
            - delivers the data to Azure;
            - deletes the delivered data.
        """
        logger.info(f"Running delivery pipeline on {self.detections_folder}..")
        metadata_csv_file_paths = get_frame_metadata_csv_file_paths(
            root_folder=self.detections_folder
        )
        logger.info(f"Number of CSVs to deliver: {len(metadata_csv_file_paths)}")

        for frame_metadata_file_path in metadata_csv_file_paths:
            self._deliver_data(frame_metadata_file_path=frame_metadata_file_path)
            self._delete_processed_data(metadata_csv_file_path=frame_metadata_file_path)

    @log_execution_time
    def _deliver_data(self, frame_metadata_file_path):
        """
        Using Azure IoT delivers the images and metadata to Azure.

        Parameters
        ----------
        frame_metadata_file_paths
            CSV files containing the metadata of the pictures,
            it's used to keep track of which files need to be delivered.

        """
        iot_handler = IoTHandler(
            hostname=self.settings["azure_iot"]["hostname"],
            device_id=self.settings["azure_iot"]["device_id"],
            shared_access_key=self.settings["azure_iot"]["shared_access_key"],
        )
        try:
            self._deliver_data_batch(frame_metadata_file_path, iot_handler)
        except FileNotFoundError as e:
            logger.error(
                f"FileNotFoundError during the delivery of: {frame_metadata_file_path}: {e}"
            )
        except Exception as e:
            logger.error(
                f"Exception during the delivery of: {frame_metadata_file_path}: {e}"
            )
            self.metadata_csv_file_paths_with_errors.append(frame_metadata_file_path)

    @log_execution_time
    def _deliver_data_batch(
        self, frame_metadata_file_path: str, iot_handler: IoTHandler
    ):
        """
        Delivers the data of a single frame metadata csv file.
        This includes the frame metadata, the detections metadata and the images containing containers.

        Returns
        -------

        """
        (
            csv_path,
            relative_path,
            detections_path,
            path_only_filtered_rows,
            file_path_only_filtered_rows,
            file_path_detection_metadata,
        ) = self._calculate_all_paths(metadata_csv_file_path=frame_metadata_file_path)

        filtered_frame_metadata_rows = []
        detection_metadata_rows = []

        with open(frame_metadata_file_path) as frame_metadata_file:
            reader = csv.reader(frame_metadata_file)
            header = next(reader)
            new_frame_metadata_header = (
                ["image_name"]
                + header
                + ["model_name", "model_version", "code_version"]
            )
            filtered_frame_metadata_rows.append(new_frame_metadata_header)
            detection_metadata_rows.append(self.detection_metadata_header)

            images_delivered = 0
            for row in reader:
                image_file_name = get_img_name_from_csv_row(csv_path, row)
                image_full_path = detections_path / image_file_name
                detection_metadata_full_path = detections_path / pathlib.Path(
                    f"{image_full_path.stem}.txt"
                )

                if os.path.isfile(image_full_path) and os.path.isfile(
                    detection_metadata_full_path
                ):
                    row_detection_metadata_rows = (
                        self._deliver_image_and_prepare_metadata(
                            image_file_name,
                            image_full_path,
                            detection_metadata_full_path,
                            iot_handler,
                        )
                    )
                    detection_metadata_rows.extend(row_detection_metadata_rows)
                    filtered_frame_metadata_rows.append(
                        [image_file_name] + row + self.model_and_code_version
                    )
                    images_delivered += 1

            if images_delivered:
                self.save_csv_file(
                    file_path_only_filtered_rows, filtered_frame_metadata_rows
                )
                upload_destination_path = f"frame_metadata/{datetime.today().strftime('%Y-%m-%d')}/{os.path.basename(file_path_only_filtered_rows)}"
                iot_handler.upload_file(
                    str(file_path_only_filtered_rows), str(upload_destination_path)
                )
                upload_destination_path = f"full_frame_metadata/{datetime.today().strftime('%Y-%m-%d')}/{os.path.basename(frame_metadata_file_path)}"
                iot_handler.upload_file(
                    str(frame_metadata_file_path), str(upload_destination_path)
                )
                self.save_csv_file(
                    file_path_detection_metadata, detection_metadata_rows
                )
                upload_destination_path = f"detection_metadata/{datetime.today().strftime('%Y-%m-%d')}/{os.path.basename(file_path_only_filtered_rows)}"
                iot_handler.upload_file(
                    str(file_path_detection_metadata), str(upload_destination_path)
                )

        logger.info(
            f"From {frame_metadata_file_path} number of frames delivered: {images_delivered}"
        )

    @staticmethod
    @log_execution_time
    def _deliver_image_and_prepare_metadata(
        image_file_name, image_full_path, detection_metadata_full_path, iot_handler
    ):
        """
        Delivers an image specified by row, and returns the metadata, this will be collected and sent all together
        for all the images of the same metadata csv file.

        Parameters
        ----------
        image_file_name:
            Filename of the image.
        image_file_name:
            full path of the image.
        detection_metadata_full_path:
            full path where the detections are located.
        iot_handler:
            IoTHandler object to deliver the data.

        Returns
        -------
        detection_metadata_rows
            Detection metadata information.

        """
        detection_metadata_rows = []

        with open(detection_metadata_full_path, "r") as detections_file:
            for detection_metadata_row in csv.reader(detections_file, delimiter=" "):
                detection_metadata_rows.append(
                    [image_file_name] + detection_metadata_row
                )

        upload_destination_path = f"images/{datetime.today().strftime('%Y-%m-%d')}/{os.path.basename(image_full_path)}"
        iot_handler.upload_file(str(image_full_path), str(upload_destination_path))
        return detection_metadata_rows

    @staticmethod
    @log_execution_time
    def save_csv_file(file_path: str, data: List[List[str]]):
        directory = os.path.dirname(file_path)
        if not os.path.exists(directory):
            os.makedirs(directory)

        with open(file_path, "w", newline="") as output_file:
            csv_writer = csv.writer(output_file)
            csv_writer.writerows(data)

    @log_execution_time
    def _delete_processed_data(self, metadata_csv_file_path):
        """
        Deletes the data that has been delivered to Azure.

        Parameters
        ----------
        metadata_csv_file_paths
            CSV files containing the metadata of the pictures,
            it's used to keep track of which files need to be delivered.
        """
        if metadata_csv_file_path not in self.metadata_csv_file_paths_with_errors:
            (
                csv_path,
                relative_path,
                detections_path,
                path_only_filtered_rows,
                file_path_only_filtered_rows,
                file_path_detection_metadata,
            ) = self._calculate_all_paths(metadata_csv_file_path=metadata_csv_file_path)

            with open(metadata_csv_file_path) as frame_metadata_file:
                reader = csv.reader(frame_metadata_file)
                _ = next(reader)
                for idx, row in enumerate(reader):
                    img_name = pathlib.Path(get_img_name_from_csv_row(csv_path, row))
                    image_full_path = detections_path / img_name
                    detection_metadata_full_path = detections_path / pathlib.Path(
                        f"{img_name.stem}.txt"
                    )
                    if os.path.isfile(image_full_path):
                        delete_file(image_full_path)
                    if os.path.isfile(detection_metadata_full_path):
                        delete_file(detection_metadata_full_path)

            if os.path.isfile(file_path_only_filtered_rows):
                delete_file(file_path_only_filtered_rows)
            if os.path.isfile(file_path_detection_metadata):
                delete_file(file_path_detection_metadata)
            if os.path.isfile(metadata_csv_file_path):
                delete_file(metadata_csv_file_path)

    def _calculate_all_paths(self, metadata_csv_file_path):
        """
        Calculate all the folders where the data should be retrieved and stored.

        Parameters
        ----------
        metadata_csv_file_path
            CSV file containing the metadata of the pictures,
            it's used to keep track of which files need to be delivered.

        Returns
        -------
            csv_path
                Path where the csv metadata file is stored. For example:
                /detections/folder1/file1.csv
            relative_path
                Folder structure to the detections folder, excluding the root. For example:
                folder1/file1.csv
            detections_path
                Path of detections excluding the file. For example:
                /detections/folder1
            path_only_filtered_rows
                Path of the metadata rows where a container has been detected. For example:
                /temp_metadata/folder1
            file_path_only_filtered_rows
                Path of the metadata rows where a container has been detected, including filename. For example:
                /temp_metadata/folder1/file1.csv
            file_path_detection_metadata
                File path of the detection metadata rows where a container has been detected. For example:
                /temp_metadata/folder1/file1-detections.csv
        """
        csv_path = pathlib.Path(metadata_csv_file_path)
        relative_path = csv_path.relative_to(self.detections_folder)
        detections_path = pathlib.Path(self.detections_folder) / relative_path.parent
        path_only_filtered_rows = (
            pathlib.Path(self.metadata_folder) / relative_path.parent
        )
        file_path_only_filtered_rows = os.path.join(
            path_only_filtered_rows, csv_path.name
        )
        file_path_detection_metadata = os.path.join(
            path_only_filtered_rows, f"{csv_path.stem}-detections.csv"
        )

        return (
            csv_path,
            relative_path,
            detections_path,
            path_only_filtered_rows,
            file_path_only_filtered_rows,
            file_path_detection_metadata,
        )
