import csv
import logging
import os
import pathlib
import shutil
import time
from typing import List

import numpy as np
import torch
from cvtoolkit.helpers.file_helpers import delete_file
from ultralytics import YOLO

from objectherkenning_openbare_ruimte.on_edge.detection_pipeline.components.input_image import (
    InputImage,
)
from objectherkenning_openbare_ruimte.on_edge.detection_pipeline.components.model_result import (
    ModelResult,
)
from objectherkenning_openbare_ruimte.on_edge.utils import (
    get_frame_metadata_csv_file_paths,
    get_img_name_from_csv_row,
    log_execution_time,
    move_file,
)
from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.
        """
        settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

        self.images_folder = pathlib.Path(settings["detection_pipeline"]["images_path"])
        self.detections_folder = settings["detection_pipeline"]["detections_path"]

        self.training_mode = settings["detection_pipeline"]["training_mode"]
        self.training_mode_destination_path = pathlib.Path(
            settings["detection_pipeline"]["training_mode_destination_path"]
        )

        self.defisheye_flag = settings["detection_pipeline"]["defisheye_flag"]
        self.defisheye_params = settings["detection_pipeline"]["defisheye_params"]

        self._check_model_availability(settings["detection_pipeline"]["sleep_time"])

        self.output_image_size = settings["detection_pipeline"]["output_image_size"]
        inference_params = settings["detection_pipeline"]["inference_params"]
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
            "project": self.detections_folder,
        }
        self.model_name = settings["detection_pipeline"]["model_name"]
        self.pretrained_model_path = os.path.join(
            settings["detection_pipeline"]["pretrained_model_path"], self.model_name
        )
        self.model = YOLO(model=self.pretrained_model_path, task="detect")
        self.target_classes = settings["detection_pipeline"]["target_classes"]
        self.sensitive_classes = settings["detection_pipeline"]["sensitive_classes"]

        self.metadata_csv_file_paths_with_errors = []

        logger.info(f"Inference_params: {self.inference_params}")
        logger.info(f"Pretrained_model_path: {self.pretrained_model_path}")
        logger.info(f"Yolo model: {self.model_name}")
        logger.info(f"Project_path: {self.detections_folder}")

    def _check_model_availability(self, sleep_time: int):
        if self.pretrained_model_path.endswith(".engine"):
            while not os.path.isfile(self.pretrained_model_path):
                logger.info(
                    f"Model {self.model_name} not found, waiting for model_conversion_pipeline.."
                )
                time.sleep(sleep_time)
        elif not os.path.isfile(self.pretrained_model_path):
            raise FileNotFoundError(f"Model not found: {self.pretrained_model_path}")

    @log_execution_time
    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects containers;
            - deletes the raw images.
        """
        logger.debug(f"Running container detection pipeline on {self.images_folder}..")
        metadata_csv_file_paths = get_frame_metadata_csv_file_paths(
            root_folder=self.images_folder
        )

        logger.info(f"Number of CSVs to detect: {len(metadata_csv_file_paths)}")
        for metadata_csv_file_path in metadata_csv_file_paths:
            self._detect_and_blur_step(metadata_csv_file_path=metadata_csv_file_path)
            if self.training_mode:
                self._move_data(metadata_csv_file_path=metadata_csv_file_path)
            elif metadata_csv_file_path not in self.metadata_csv_file_paths_with_errors:
                self._delete_data_step(metadata_csv_file_path=metadata_csv_file_path)

    @log_execution_time
    def _detect_and_blur_step(self, metadata_csv_file_path):
        """
        Loops through each row of the metadata csv file, detects containers and blur each image.

        Parameters
        ----------
        metadata_csv_file_path : str
            Metadata csv file path.
        """
        try:
            logger.debug(f"metadata_csv_file_path: {metadata_csv_file_path}")
            (
                csv_path,
                _,
                images_path,
                detections_path,
            ) = self._calculate_all_paths(metadata_csv_file_path)

            with open(metadata_csv_file_path) as frame_metadata_file:
                reader = csv.reader(frame_metadata_file)
                _ = next(reader)
                processed_images_count = target_objects_detected_count = 0
                for row in reader:
                    image_file_name = pathlib.Path(
                        get_img_name_from_csv_row(csv_path, row)
                    )
                    image_full_path = images_path / image_file_name
                    if os.path.isfile(image_full_path):
                        target_objects_detected_count += self._detect_and_blur_image(
                            image_file_name,
                            image_full_path,
                            csv_path,
                            detections_path,
                        )
                        processed_images_count += 1
                    else:
                        logger.debug(f"Image {image_full_path} not found, skipping.")
            if target_objects_detected_count:
                shutil.copyfile(csv_path, os.path.join(detections_path, csv_path.name))
            if metadata_csv_file_path in self.metadata_csv_file_paths_with_errors:
                self.metadata_csv_file_paths_with_errors.remove(metadata_csv_file_path)
            logger.info(
                f"Processed {processed_images_count} images from {metadata_csv_file_path}, "
                f"detected {target_objects_detected_count} containers."
            )
        except Exception as e:
            logger.error(
                f"Exception during the detection of: {metadata_csv_file_path}: {e}"
            )
            self.metadata_csv_file_paths_with_errors.append(metadata_csv_file_path)

    @log_execution_time
    def _delete_data_step(self, metadata_csv_file_path):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        metadata_csv_file_path
            Path of the CSV file containing the metadata of the pictures,
            it's used to keep track of which files needs to be deleted.
        """
        csv_path = pathlib.Path(metadata_csv_file_path)
        relative_path = csv_path.relative_to(self.images_folder)
        images_path = self.images_folder / relative_path.parent
        with open(metadata_csv_file_path) as frame_metadata_file:
            images_deleted_count = 0
            reader = csv.reader(frame_metadata_file)
            _ = next(reader)
            for idx, row in enumerate(reader):
                image_file_name = get_img_name_from_csv_row(csv_path, row)
                image_full_path = images_path / image_file_name
                if os.path.isfile(image_full_path):
                    delete_file(image_full_path)
                    images_deleted_count += 1
        delete_file(metadata_csv_file_path)
        logger.info(
            f"Deleted {images_deleted_count} images from {metadata_csv_file_path}"
        )

    @log_execution_time
    def _move_data(self, metadata_csv_file_path):
        """
        Moves the data that has been processed to a training folder.

        Parameters
        ----------
        metadata_csv_file_path
            Path of the CSV file containing the metadata of the pictures,
            it's used to keep track of which files had to be detected.
        """
        csv_path = pathlib.Path(metadata_csv_file_path)
        relative_path = csv_path.relative_to(self.images_folder)
        images_path = self.images_folder / relative_path.parent
        with open(metadata_csv_file_path) as frame_metadata_file:
            images_moved_count = 0
            reader = csv.reader(frame_metadata_file)
            _ = next(reader)
            for idx, row in enumerate(reader):
                image_file_name = get_img_name_from_csv_row(csv_path, row)
                image_full_path = images_path / image_file_name
                if os.path.isfile(image_full_path):
                    image_subfolder_and_name = os.path.relpath(
                        image_full_path, self.images_folder
                    )
                    image_destination_full_path = os.path.join(
                        self.training_mode_destination_path,
                        image_subfolder_and_name,
                    )
                    move_file(image_full_path, image_destination_full_path)
                    images_moved_count += 1
        metadata_csv_subfolder_and_name = os.path.relpath(csv_path, self.images_folder)
        metadata_csv_destination_file_path = os.path.join(
            self.training_mode_destination_path, metadata_csv_subfolder_and_name
        )
        move_file(csv_path, metadata_csv_destination_file_path)
        logger.info(
            f"Training mode on: Moved {images_moved_count} images from {metadata_csv_file_path}"
        )

    @log_execution_time
    def _detect_and_blur_image(
        self,
        image_file_name: pathlib.Path,
        image_full_path: pathlib.Path,
        csv_path: pathlib.Path,
        detections_path: pathlib.Path,
    ):
        """Loads the image, resizes it, detects containers and blur sensitive data.

        Parameters
        ----------
        image_file_name : pathlib.Path
            File name of the image
        image_full_path : pathlib.Path
            Path of the image
        csv_path : pathlib.Path
            Path where the csv metadata file is stored. For example:
            /detections/folder1/file1.csv
        detections_path : pathlib.Path
            Path of detections excluding the file. For example:
            /detections/folder1

        Returns
        -------
        int
            Count of detected target objects
        """
        logger.debug(f"Detecting and blurring: {image_file_name}")

        image = InputImage(image_full_path=str(image_full_path))
        image.resize(output_image_size=self.output_image_size)

        if self.defisheye_flag:
            image.defisheye(defisheye_params=self.defisheye_params)

        self.inference_params["source"] = image
        self.inference_params["name"] = csv_path.stem
        detection_results = self.model(**self.inference_params)
        torch.cuda.empty_cache()

        self._process_detections_and_blur_image(
            detection_results,
            str(detections_path),
            image_file_name,
        )
        return sum(
            len(
                np.where(
                    np.in1d(
                        model_result.cpu().boxes.numpy().cls,
                        self.target_classes,
                    )
                )[0]
            )
            for model_result in detection_results
        )

    @log_execution_time
    def _process_detections_and_blur_image(
        self,
        model_results: List,
        image_detection_path: str,
        image_file_name: pathlib.Path,
    ):
        for model_result in model_results:
            model_result = ModelResult(
                model_result,
                target_classes=self.target_classes,
                sensitive_classes=self.sensitive_classes,
            )
            model_result.process_detections_and_blur_sensitive_data(
                image_detection_path=image_detection_path,
                image_file_name=image_file_name,
            )

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
                Folder structure to the images folder, excluding the root. For example:
                folder1/file1.csv
            images_path
                Path of images excluding the file. For example:
                /detections/folder1
            detections_path
                Path of detections excluding the file. For example:
                /detections/folder1
        """
        csv_path = pathlib.Path(metadata_csv_file_path)
        relative_path = csv_path.relative_to(self.images_folder)
        images_path = self.images_folder / relative_path.parent
        detections_path = pathlib.Path(self.detections_folder) / relative_path.parent
        return (
            csv_path,
            relative_path,
            images_path,
            detections_path,
        )
