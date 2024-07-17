import logging
import os
import secrets
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import numpy as np
import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from objectherkenning_openbare_ruimte.inference_pipeline.source import blurring_tools

logger = logging.getLogger(__name__)


class DataInference:
    def __init__(
        self,
        images_folder: str,
        inference_folder: str,
        model_name: str,
        pretrained_model_path: str,
        inference_params: Dict,
        target_classes: List,
        sensitive_classes: List,
        output_image_size: Tuple[int, int],
        save_detections: bool,
        save_labels: bool,
        defisheye_flag: bool,
        defisheye_params: Dict,
        batch_size: int = 1,
    ):
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.images_folder = images_folder
        self.inference_folder = inference_folder
        self.model_name = model_name
        self.pretrained_model_path = pretrained_model_path
        self.inference_params = inference_params
        logger.debug(f"Inference_params: {self.inference_params}")
        logger.debug(f"Pretrained_model_path: {self.pretrained_model_path}")
        logger.debug(f"Yolo model: {self.model_name}")
        logger.debug(f"Project_path: {self.inference_folder}")

        self.model = YOLO(model=self.pretrained_model_path, task="detect")

        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.output_image_size = output_image_size
        self.save_detections = save_detections
        self.save_labels = save_labels

        self.defisheye_flag = defisheye_flag
        self.defisheye_params = defisheye_params
        self.mapx = self.mapy = None

        self.batch_size = batch_size

        # Predefined colors for 5 categories
        predefined_colors = {
            0: (0, 0, 255),  # Blue
            1: (0, 255, 0),  # Green
            2: (255, 0, 0),  # Red
            3: (255, 255, 0),  # Cyan
            4: (255, 0, 255),  # Magenta
        }

        # Initialize the category_colors dictionary with predefined colors and add random colors for new categories
        self.category_colors = defaultdict(
            lambda: (
                secrets.randbelow(256),
                secrets.randbelow(256),
                secrets.randbelow(256),
            ),
            predefined_colors,
        )

    def run_pipeline(self):
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects everything;
            - stores labels if required;
            - stores images if required, with
              - sensitive classes blurred;
              - target classes bounding boxes drawn;
        """
        logger.debug(f"Running detection pipeline on {self.images_folder}..")
        folders_and_frames = self._find_image_paths_and_group_by_folder(
            root_folder=self.images_folder
        )
        logger.debug(
            f"Number of images to detect: {sum(len(frames) for frames in folders_and_frames.values())}"
        )
        self._detect(folders_and_frames=folders_and_frames)

    def _detect(self, folders_and_frames: Dict[str, List[str]]):
        results = self._process_batches(
            self.model,
            folders_and_frames,
            batch_size=self.batch_size,
        )
        logger.debug(results)

    def _load_and_defisheye_image(self, image_path):
        image = cv2.resize(cv2.imread(str(image_path)), self.output_image_size)
        if self.defisheye_flag:
            image = self._defisheye(image)
        return image

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

    def _process_batches(self, model, folders_and_frames, batch_size):
        for folder_name, images in folders_and_frames.items():
            images_paths = [os.path.join(folder_name, image) for image in images]
            processed_images = 0
            for i in range(0, len(images_paths), batch_size):
                try:
                    batch_images = [
                        self._load_and_defisheye_image(image_path)
                        for image_path in images_paths[i : i + batch_size]
                    ]
                    self.inference_params["source"] = batch_images
                    self.inference_params["name"] = folder_name
                    batch_results = model(**self.inference_params)
                    torch.cuda.empty_cache()  # Clear unused memory
                    self._process_results(
                        batch_results, images_paths[i : i + batch_size]
                    )
                    processed_images += len(batch_images)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.debug(
                            "Out of memory with batch size of {}".format(batch_size)
                        )
                        if batch_size > 1:
                            new_batch_size = batch_size // 2
                            logger.debug(
                                "Trying smaller batch size: {}".format(new_batch_size)
                            )
                            return self._process_batches(
                                model, images_paths, new_batch_size
                            )
                        else:
                            raise RuntimeError(
                                "Out of memory with the smallest batch size"
                            )
                    else:
                        raise e
            logger.debug(f"Number of images processed: {processed_images}")

    def _process_results(self, model_results: Results, images_paths: List):

        for r, image_path in zip(model_results, images_paths):
            result = r.cpu()
            boxes = result.boxes.numpy()

            image_name = os.path.basename(image_path)
            subfolder = os.path.relpath(os.path.dirname(image_path), self.images_folder)

            logger.debug(f"=== IMAGE: {subfolder}/{image_name} ===")

            target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
            if len(target_idxs) == 0:  # Nothing to do!
                logger.debug("No target classes detected.")
                continue

            image = result.orig_img.copy()
            sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

            # Get categories for each bounding box
            categories = [int(box.cls) for box in boxes]

            logger.debug(f"=== CATEGORIES: {categories} ===")

            # Blur sensitive data
            sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
            image = blurring_tools.blur_inside_boxes(image, sensitive_bounding_boxes)

            # Draw annotation boxes
            target_bounding_boxes = boxes[target_idxs].xyxy
            target_categories = [int(box.cls) for box in boxes[target_idxs]]
            image = blurring_tools.draw_bounding_boxes(
                image, target_bounding_boxes, target_categories, self.category_colors
            )

            # Save image
            if self.save_detections:
                images_dir = os.path.join(
                    self.inference_folder, "detected_images", subfolder
                )
                os.makedirs(images_dir, exist_ok=True)
                save_path = os.path.join(images_dir, image_name)
                cv2.imwrite(save_path, image)
                logger.debug("=== SAVED IMAGE ===")

            # Save annotation
            if self.save_labels:
                annotation_str = self._get_annotion_string_from_boxes(boxes)
                labels_dir = os.path.join(
                    self.inference_folder, "detected_labels", subfolder
                )
                os.makedirs(labels_dir, exist_ok=True)
                annotation_path = os.path.join(
                    labels_dir, f"{os.path.splitext(image_name)[0]}.txt"
                )
                with open(annotation_path, "w") as f:
                    f.write(annotation_str)
                logger.debug("=== SAVED LABELS ===")

    @staticmethod
    def _get_annotion_string_from_boxes(boxes: Boxes) -> str:
        boxes = boxes.cpu()

        annotation_lines = []

        for box in boxes:
            cls = int(box.cls.squeeze())
            conf = float(box.conf.squeeze())
            tracking_id = int(box.id.squeeze()) if box.is_track else -1
            yolo_box_str = " ".join([f"{x:.6f}" for x in box.xywhn.squeeze()])
            annotation_lines.append(f"{cls} {yolo_box_str} {conf:.6f} {tracking_id}")
        return "\n".join(annotation_lines)

    @staticmethod
    def _find_image_paths_and_group_by_folder(root_folder):
        folders_and_frames = {}
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    if foldername not in folders_and_frames:
                        folders_and_frames[foldername] = [filename]
                    else:
                        folders_and_frames[foldername].append(filename)
        return folders_and_frames
