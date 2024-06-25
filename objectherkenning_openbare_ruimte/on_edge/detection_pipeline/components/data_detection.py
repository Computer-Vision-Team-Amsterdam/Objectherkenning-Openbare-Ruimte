import csv
import logging
import os
import pathlib
import shutil
from typing import Dict, List, Tuple

import cv2
import numpy as np
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
)

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
        images_folder: str,
        detections_folder: str,
        model_name: str,
        pretrained_model_path: str,
        output_image_size: Tuple[int, int],
        inference_params: Dict,
        defisheye_flag: bool,
        defisheye_params: Dict,
        target_classes: List,
        sensitive_classes: List,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.images_folder = images_folder
        self.detections_folder = detections_folder
        self.model_name = model_name
        self.pretrained_model_path = os.path.join(pretrained_model_path, model_name)
        self.output_image_size = output_image_size
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

        self.model = YOLO(model=self.pretrained_model_path, task="detect")
        self.roi = self.mapx = self.mapy = None
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes

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
        self._detect_and_blur(metadata_csv_file_paths=metadata_csv_file_paths)
        self._delete_data(metadata_csv_file_paths=metadata_csv_file_paths)

    def _detect_and_blur(self, metadata_csv_file_paths):
        for metadata_csv_file_path in metadata_csv_file_paths:
            logger.debug(f"metadata_csv_file_path: {metadata_csv_file_path}")
            csv_path = pathlib.Path(metadata_csv_file_path)
            relative_path = csv_path.relative_to(self.images_folder)
            images_path = pathlib.Path(self.images_folder) / relative_path.parent
            detections_path = (
                pathlib.Path(self.detections_folder) / relative_path.parent
            )

            with open(metadata_csv_file_path) as frame_metadata_file:
                reader = csv.reader(frame_metadata_file)
                _ = next(reader)
                processed_images_count = target_objects_detected_count = 0
                for idx, row in enumerate(reader):
                    image_file_name = pathlib.Path(
                        get_img_name_from_csv_row(csv_path, row)
                    )
                    image_full_path = images_path / image_file_name
                    if os.path.isfile(image_full_path):
                        logger.info(f"Processing {image_file_name}")
                        image = cv2.imread(str(image_full_path))
                        if self.defisheye_flag:
                            image = self._defisheye(image)
                        image = cv2.resize(image, self.output_image_size)
                        self.inference_params["source"] = image
                        self.inference_params["name"] = csv_path.stem

                        detection_results = self.model(**self.inference_params)
                        torch.cuda.empty_cache()

                        target_objects_detected_count += sum(
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

                        self._process_results(
                            detection_results,
                            str(detections_path),
                            image_file_name,
                        )
                        processed_images_count += 1
                    else:
                        logger.debug(f"Image {image_full_path} not found, skipping.")
            if target_objects_detected_count:
                shutil.copyfile(csv_path, os.path.join(detections_path, csv_path.name))
            logger.info(
                f"Processed {processed_images_count} images from {metadata_csv_file_path}, "
                f"detected {target_objects_detected_count} containers."
            )

    def _defisheye(self, image):
        if self.roi is None or self.mapx is None or self.mapy is None:
            newcameramtx, self.roi = cv2.getOptimalNewCameraMatrix(
                np.array(self.defisheye_params["camera_matrix"]),
                np.array(self.defisheye_params["distortion_params"]),
                self.defisheye_params["input_image_size"],
                1,
                self.defisheye_params["input_image_size"],
            )
            self.mapx, self.mapy = cv2.initUndistortRectifyMap(
                np.array(self.defisheye_params["camera_matrix"]),
                np.array(self.defisheye_params["distortion_params"]),
                None,
                newcameramtx,
                self.defisheye_params["input_image_size"],
                5,
            )

        # undistort
        img_dst = cv2.remap(image, self.mapx, self.mapy, cv2.INTER_LINEAR)

        # crop the image
        x, y, w, h = self.roi
        img_dst = img_dst[y : y + h, x : x + w]

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

    def _delete_data(self, metadata_csv_file_paths):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        videos_and_frames
            List containing the paths of the images to delete.
        """
        for metadata_csv_file_path in metadata_csv_file_paths:
            csv_path = pathlib.Path(metadata_csv_file_path)
            relative_path = csv_path.relative_to(self.images_folder)
            images_path = pathlib.Path(self.images_folder) / relative_path.parent
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
