import csv
import logging
import os
import pathlib
import shutil
import time
from typing import Dict, List, Tuple

import cv2
import numpy as np
import pyvips
import torch
from cvtoolkit.helpers.file_helpers import delete_file
from ultralytics import YOLO
from ultralytics.engine.results import Boxes

from objectherkenning_openbare_ruimte.on_edge.detection_pipeline.components import (
    blurring_tools,
)
from objectherkenning_openbare_ruimte.on_edge.utils import (
    get_frame_metadata_csv_file_paths,
    get_img_name_from_csv_row,
    log_execution_time,
    move_file,
)

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
        images_folder: str,
        detections_folder: str,
        model_name: str,
        pretrained_model_path: str,
        wait_for_model_timeout: float,
        input_image_size: Tuple[int, int],
        output_image_size: Tuple[int, int],
        inference_params: Dict,
        defisheye_flag: bool,
        defisheye_params: Dict,
        target_classes: List,
        sensitive_classes: List,
        training_mode: bool,
        training_mode_destination_path: str,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.training_mode = training_mode
        self.training_mode_destination_path = pathlib.Path(
            training_mode_destination_path
        )
        self.images_folder = pathlib.Path(images_folder)
        self.detections_folder = detections_folder
        self.model_name = model_name
        self.pretrained_model_path = os.path.join(pretrained_model_path, model_name)
        self.input_image_size = input_image_size
        self.output_image_size = output_image_size
        self.shrink_factor = None
        self.resize_backend = "pyvips"
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
            "project": self.detections_folder,
        }
        self.defisheye_flag = defisheye_flag
        self.defisheye_params = defisheye_params
        logger.info(f"Inference_params: {self.inference_params}")
        logger.info(f"Pretrained_model_path: {self.pretrained_model_path}")
        logger.info(f"Yolo model: {self.model_name}")
        logger.info(f"Project_path: {self.detections_folder}")

        if self.pretrained_model_path.endswith(".engine"):
            while not os.path.isfile(self.pretrained_model_path):
                logger.info(
                    f"Model {self.model_name} not found, waiting for model_conversion_pipeline.."
                )
                time.sleep(wait_for_model_timeout)
        elif not os.path.isfile(self.pretrained_model_path):
            raise FileNotFoundError(f"Model not found: {self.pretrained_model_path}")
        self.model = YOLO(model=self.pretrained_model_path, task="detect")
        self.mapx = self.mapy = None
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.metadata_csv_file_paths_with_errors: List[str] = []

    @log_execution_time
    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects containers;
            - deletes the raw images.
        """
        logger.info(f"Running container detection pipeline on {self.images_folder}..")
        metadata_csv_file_paths = get_frame_metadata_csv_file_paths(
            root_folder=self.images_folder
        )
        logger.info(f"Number of CSVs to detect: {len(metadata_csv_file_paths)}")
        self._detect_and_blur_step(metadata_csv_file_paths=metadata_csv_file_paths)
        if self.training_mode:
            self._move_data(metadata_csv_file_paths=metadata_csv_file_paths)
        else:
            self._delete_data_step(metadata_csv_file_paths=metadata_csv_file_paths)

    @log_execution_time
    def _detect_and_blur_step(self, metadata_csv_file_paths):
        for metadata_csv_file_path in metadata_csv_file_paths:
            try:
                logger.debug(f"metadata_csv_file_path: {metadata_csv_file_path}")
                (
                    csv_path,
                    relative_path,
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
                            target_objects_detected_count += (
                                self._detect_and_blur_image(
                                    image_file_name,
                                    image_full_path,
                                    csv_path,
                                    detections_path,
                                )
                            )
                            processed_images_count += 1
                        else:
                            logger.debug(
                                f"Image {image_full_path} not found, skipping."
                            )
                if target_objects_detected_count:
                    shutil.copyfile(
                        csv_path, os.path.join(detections_path, csv_path.name)
                    )
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
    def _detect_and_blur_image(
        self, image_file_name, image_full_path, csv_path, detections_path
    ):
        logger.info(f"Detecting and blurring: {image_file_name}")
        image = self._load_and_resize(image_full_path)
        if self.defisheye_flag:
            image = self._defisheye(image)

        self.inference_params["source"] = image
        self.inference_params["name"] = csv_path.stem

        detection_results = self.model(**self.inference_params)
        torch.cuda.empty_cache()

        self._process_results(
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

    @log_execution_time
    def _load_and_resize_WIP(self, image_full_path):
        # TODO:
        # - shrink factor for pyvips can only be a multiple of 2
        # - pyvips shrinking is block shrinking, can cause artefacts
        # - first pyvips shrink by 2 and then opencv / other method for refining?
        if self.shrink_factor is None:
            shrink_factors = (
                self.input_image_size[0] / self.output_image_size[0],
                self.input_image_size[1] / self.output_image_size[1],
            )
            if shrink_factors[0] != shrink_factors[1]:
                raise ValueError(
                    "Invalid output_image_size dimensions: aspect ratio should be preserved"
                )
            if shrink_factors[0] != int(shrink_factors[0]):
                self.shrink_factor = 1
                self.resize_backend = "opencv"
                logger.debug(
                    f"Non-integer shrink factor {shrink_factors[0]} for {self.input_image_size} -> {self.output_image_size}."
                    f" Using {self.resize_backend} for loading and resizing images. This may reduce performance."
                )
            else:
                self.shrink_factor = int(shrink_factors[0])
                self.resize_backend = "pyvips"
                logger.debug(
                    f"Using {self.resize_backend} for loading and resizing images with shrink factor {self.shrink_factor}."
                )

        if self.resize_backend == "pyvips":
            image = pyvips.Image.new_from_file(
                str(image_full_path), access="sequential", shrink=self.shrink_factor
            )
            return image.numpy()[:, :, ::-1]
        elif self.resize_backend == "opencv":
            image = cv2.imread(str(image_full_path))
            return cv2.resize(image, self.output_image_size)

    @log_execution_time
    def _load_and_resize(self, image_full_path):
        image = cv2.imread(str(image_full_path))
        return cv2.resize(image, self.output_image_size)

    @log_execution_time
    def _defisheye(self, image):
        if self.mapx is None or self.mapy is None:
            old_w, old_h = self.defisheye_params["input_image_size"]
            new_h, new_w = image.shape[:2]

            cam_mtx = np.array(self.defisheye_params["camera_matrix"])
            cam_mtx[0, :] = cam_mtx[0, :] * (float(new_w) / float(old_w))
            cam_mtx[1, :] = cam_mtx[1, :] * (float(new_h) / float(old_h))

            logger.debug(f"Defisheye: {(old_w, old_h)} -> ({(new_w, new_h)})")
            logger.debug(f"Scaled camera matrix:\n{cam_mtx}")

            newcameramtx, _ = cv2.getOptimalNewCameraMatrix(
                cam_mtx,
                np.array(self.defisheye_params["distortion_params"]),
                (new_w, new_h),
                0,
                (new_w, new_h),
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                cam_mtx,
                np.array(self.defisheye_params["distortion_params"]),
                None,
                newcameramtx,
                (new_w, new_h),
                5,
            )

        # undistort
        img_dst = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        return img_dst

    @staticmethod
    def _get_annotation_string_from_boxes(boxes: Boxes) -> str:
        boxes = boxes.cpu()
        annotation_lines = []

        for box in boxes:
            cls = int(box.cls.squeeze())
            conf = float(box.conf.squeeze())
            tracking_id = int(box.id.squeeze()) if box.is_track else -1
            yolo_box_str = " ".join([f"{x:.6f}" for x in box.xywhn.squeeze()])
            annotation_lines.append(f"{cls} {yolo_box_str} {conf:.6f} {tracking_id}")
        return "\n".join(annotation_lines)

    @log_execution_time
    def _process_results(
        self,
        model_results: List,
        image_detection_path: str,
        image_file_name: pathlib.Path,
    ):
        for model_result in model_results:
            result = model_result.cpu()
            boxes = result.boxes.numpy()

            for summary_str in self._yolo_result_summary(result):
                logger.info(summary_str)

            target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
            # logger.debug(f"target_idxs {target_idxs}")
            if len(target_idxs) == 0:  # Nothing to do!
                logger.debug("No container detected, not storing the image.")
                return False

            image = result.orig_img.copy()
            sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

            if len(sensitive_idxs) > 0:
                # Blur sensitive data
                sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
                image = blurring_tools.blur_inside_boxes(
                    image, sensitive_bounding_boxes
                )

            # Draw annotation boxes
            target_bounding_boxes = boxes[target_idxs].xyxy
            image = blurring_tools.draw_bounding_boxes(image, target_bounding_boxes)

            logger.debug(f"Folder path: {image_detection_path}")
            pathlib.Path(image_detection_path).mkdir(parents=True, exist_ok=True)
            logger.debug(f"Save path: {image_detection_path}")
            result_full_path = os.path.join(image_detection_path, image_file_name)
            cv2.imwrite(result_full_path, image)
            logger.debug("Saved image.")

            # Save annotation
            annotation_str = self._get_annotation_string_from_boxes(boxes[target_idxs])
            annotation_path = os.path.join(
                image_detection_path, f"{image_file_name.stem}.txt"
            )
            with open(annotation_path, "w") as f:
                f.write(annotation_str)

            return True

    def _yolo_result_summary(self, result):
        obj_classes, obj_counts = np.unique(result.boxes.cls, return_counts=True)
        obj_str = "Detected: {"
        for obj_cls, obj_count in zip(obj_classes, obj_counts):
            obj_str = obj_str + f"{result.names[obj_cls]}: {obj_count}, "
        if len(obj_classes):
            obj_str = obj_str[0:-2]
        obj_str = obj_str + "}"

        speed_str = "Compute: {"
        for key, value in result.speed.items():
            speed_str = speed_str + f"{key}: {value:.2f}ms, "
        speed_str = speed_str[0:-2] + "}"

        return [obj_str, speed_str]

    @log_execution_time
    def _delete_data_step(self, metadata_csv_file_paths):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        metadata_csv_file_paths
            CSV files containing the metadata of the pictures,
            it's used to keep track of which files had to be detected.
        """
        for metadata_csv_file_path in list(
            set(metadata_csv_file_paths) - set(self.metadata_csv_file_paths_with_errors)
        ):
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
    def _move_data(self, metadata_csv_file_paths):
        """
        Moves the data that has been processed to a training folder.

        Parameters
        ----------
        metadata_csv_file_paths
            CSV files containing the metadata of the pictures,
            it's used to keep track of which files had to be detected.
        """
        for metadata_csv_file_path in metadata_csv_file_paths:
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
            metadata_csv_subfolder_and_name = os.path.relpath(
                csv_path, self.images_folder
            )
            metadata_csv_destination_file_path = os.path.join(
                self.training_mode_destination_path, metadata_csv_subfolder_and_name
            )
            move_file(csv_path, metadata_csv_destination_file_path)
            logger.info(
                f"Moved {images_moved_count} images from {metadata_csv_file_path}"
            )
