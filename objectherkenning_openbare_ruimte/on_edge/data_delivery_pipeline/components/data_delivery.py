import csv
import logging
import os
import pathlib

from cvtoolkit.helpers.file_helpers import delete_file

from objectherkenning_openbare_ruimte.on_edge.data_delivery_pipeline.components.iot_handler import (
    IoTHandler,
)
from objectherkenning_openbare_ruimte.on_edge.utils import (
    get_frame_metadata_csv_file_paths,
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
        logger.info(f"Running delivery pipeline on {self.detections_folder}..")
        metadata_csv_file_paths = get_frame_metadata_csv_file_paths(
            root_folder=self.detections_folder
        )
        logger.info(f"Number of CSVs to deliver: {len(metadata_csv_file_paths)}")
        self._deliver_data(metadata_csv_file_paths=metadata_csv_file_paths)
        self._delete_data(metadata_csv_file_paths=metadata_csv_file_paths)

    def _calculate_all_paths(self, metadata_csv_file_path):
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

    def _deliver_data(self, metadata_csv_file_paths):
        """
        Using Azure IoT delivers the images and metadata to Azure.

        Parameters
        ----------
        metadata_csv_file_paths

        """
        iot_handler = IoTHandler(
            hostname=self.iot_settings["hostname"],
            device_id=self.iot_settings["device_id"],
            shared_access_key=self.iot_settings["shared_access_key"],
        )
        for metadata_csv_file_path in metadata_csv_file_paths:
            (
                csv_path,
                relative_path,
                detections_path,
                path_only_filtered_rows,
                file_path_only_filtered_rows,
                file_path_detection_metadata,
            ) = self._calculate_all_paths(metadata_csv_file_path=metadata_csv_file_path)
            logger.debug(f"metadata_csv_file_path: {metadata_csv_file_path}")
            logger.debug(f"csv_path: {csv_path}")
            logger.debug(f"relative_path: {relative_path}")
            logger.debug(f"detections_path: {detections_path}")
            logger.debug(f"path_only_filtered_rows: {path_only_filtered_rows}")
            logger.debug(
                f"file_path_only_filtered_rows: {file_path_only_filtered_rows}"
            )
            logger.debug(
                f"file_path_detection_metadata: {file_path_detection_metadata}"
            )

            filtered_rows = []
            detection_metadata_rows = []
            detection_metadata_header = [
                "image_name",
                "x_center",
                "y_center",
                "width",
                "height",
                "confidence",
                "tracking_id",
            ]
            try:
                with open(metadata_csv_file_path) as frame_metadata_file:
                    reader = csv.reader(frame_metadata_file)
                    header = next(reader)
                    new_header = (
                        ["image_name"]
                        + header
                        + ["model_name", "model_version", "code_version"]
                    )
                    filtered_rows.append(new_header)
                    detection_metadata_rows.append(detection_metadata_header)
                    images_delivered = 0
                    for idx, row in enumerate(reader):
                        image_file_name = pathlib.Path(f"{csv_path.stem}-{row[1]}.png")
                        detection_metadata_file_name = pathlib.Path(
                            f"{csv_path.stem}-{row[1]}.txt"
                        )
                        image_full_path = detections_path / image_file_name
                        detection_metadata_full_path = (
                            detections_path / detection_metadata_file_name
                        )
                        if os.path.isfile(image_full_path) and os.path.isfile(
                            detection_metadata_full_path
                        ):
                            new_row = [image_file_name] + row
                            filtered_rows.append(new_row)
                            iot_handler.upload_file(str(image_full_path))
                            logger.debug(f"Delivered: {str(image_full_path)}")
                            with open(
                                detection_metadata_full_path, "r"
                            ) as detections_file:
                                detections_reader = csv.reader(
                                    detections_file, delimiter=" "
                                )
                                detection_metadata_rows.extend(list(detections_reader))
                            images_delivered += 1
                    if images_delivered:
                        logger.debug(
                            f"From {metadata_csv_file_path} number of frames delivered: {images_delivered}"
                        )
                        logger.debug(f"filtered_rows: {filtered_rows}")
                        logger.debug(
                            f"detection_metadata_rows: {detection_metadata_rows}"
                        )
                        if not os.path.exists(path_only_filtered_rows):
                            os.makedirs(path_only_filtered_rows)
                        with open(
                            file_path_only_filtered_rows, "w", newline=""
                        ) as output_file:
                            csv_writer = csv.writer(output_file)
                            csv_writer.writerows(filtered_rows)
                        iot_handler.upload_file(str(file_path_only_filtered_rows))
                        with open(
                            file_path_detection_metadata, "w", newline=""
                        ) as output_file:
                            csv_writer = csv.writer(output_file)
                            csv_writer.writerows(detection_metadata_rows)
                        iot_handler.upload_file(str(file_path_detection_metadata))

                logger.info(
                    f"From {metadata_csv_file_path} number of frames delivered: {images_delivered}"
                )
            except FileNotFoundError as e:
                logger.error(
                    f"FileNotFoundError during the delivery of: {metadata_csv_file_path}: {e}"
                )
            except Exception as e:
                logger.error(
                    f"Exception during the delivery of: {metadata_csv_file_path}: {e}"
                )

    def _delete_data(self, metadata_csv_file_paths):
        """
        Deletes the data that has been delivered to Azure.

        Parameters
        ----------
        metadata_csv_file_paths

        """
        for metadata_csv_file_path in metadata_csv_file_paths:
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
                    image_file_name = pathlib.Path(f"{csv_path.stem}-{row[1]}.png")
                    detection_metadata_file_name = pathlib.Path(
                        f"{csv_path.stem}-{row[1]}.txt"
                    )
                    image_full_path = detections_path / image_file_name
                    detection_metadata_full_path = (
                        detections_path / detection_metadata_file_name
                    )
                    if os.path.isfile(image_full_path):
                        delete_file(image_full_path)
                        logger.debug(f"Deleted: {image_full_path}")
                    if os.path.isfile(detection_metadata_full_path):
                        delete_file(detection_metadata_full_path)
                        logger.debug(f"Deleted: {detection_metadata_full_path}")
            delete_file(file_path_only_filtered_rows)
            logger.debug(f"Deleted: {file_path_only_filtered_rows}")
            delete_file(file_path_detection_metadata)
            logger.debug(f"Deleted: {file_path_detection_metadata}")
            delete_file(metadata_csv_file_path)
            logger.debug(f"Deleted: {metadata_csv_file_path}")
