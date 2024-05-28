import logging
import os
import pathlib
from typing import Dict, List

import cv2
import numpy as np
import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS, delete_file
from ultralytics import YOLO
from ultralytics.engine.results import Boxes, Results

from objectherkenning_openbare_ruimte.detection_pipeline.components import (
    blurring_tools,
)

logger = logging.getLogger("detection_pipeline")


class DataDetection:
    def __init__(
        self,
        images_folder: str,
        detections_folder: str,
        model_name: str,
        pretrained_model_path: str,
        inference_params: Dict,
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
        self.inference_params = {
            "imgsz": inference_params.get("img_size", 640),
            "save": inference_params.get("save_img_flag", False),
            "save_txt": inference_params.get("save_txt_flag", False),
            "save_conf": inference_params.get("save_conf_flag", False),
            "conf": inference_params.get("conf", 0.25),
            "project": self.detections_folder,
        }
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
        videos_and_frames = self._find_image_paths_and_group_by_videoname(
            root_folder=self.images_folder
        )
        logger.info(
            f"Number of images to detect: {sum(len(frames) for frames in videos_and_frames.values())}"
        )
        self._detect_containers(videos_and_frames=videos_and_frames)
        self._delete_data(videos_and_frames=videos_and_frames)

    @staticmethod
    def _find_image_paths_and_group_by_videoname(root_folder):
        videos_and_frames = {}
        for foldername, subfolders, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    video_name, _ = os.path.basename(filename).rsplit("_frame_", 1)
                    image_path = os.path.join(foldername, filename)
                    if video_name not in videos_and_frames:
                        videos_and_frames[video_name] = [image_path]
                    else:
                        videos_and_frames[video_name].append(image_path)
        return videos_and_frames

    @staticmethod
    def _delete_data(videos_and_frames: Dict[str, List[str]]):
        """
        Deletes the data that has been processed.

        Parameters
        ----------
        videos_and_frames
            List containing the paths of the images to delete.
        """
        batch_count = 0
        for video_name, images_paths in videos_and_frames.items():
            for image_path in images_paths:
                delete_file(image_path)
                batch_count += 1
        logger.info(f"Number of images deleted: {batch_count}")

    def _detect_containers(self, videos_and_frames: Dict[str, List[str]]):
        results = self._process_batches(self.model, videos_and_frames, batch_size=1)
        logger.info(results)

    def _process_batches(self, model, videos_and_frames, batch_size):
        for video_name, images_paths in videos_and_frames.items():
            self.inference_params["source"] = images_paths
            processed_images = 0
            for i in range(0, len(images_paths), batch_size):
                try:
                    batch_images = [
                        cv2.imread(image_path)
                        for image_path in images_paths[i : i + batch_size]
                    ]
                    # batch_images = self._defisheye(batch_images)
                    self.inference_params["source"] = batch_images
                    self.inference_params["name"] = video_name + "_batch_"
                    batch_results = model(**self.inference_params)
                    logger.debug(f"Result YOLO: {batch_results}")
                    torch.cuda.empty_cache()  # Clear unused memory
                    self._process_results(batch_results)
                    processed_images += len(batch_images)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(
                            "Out of memory with batch size of {}".format(batch_size)
                        )
                        if batch_size > 1:
                            new_batch_size = batch_size // 2
                            logger.warning(
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
            logger.info(f"Number of images processed: {processed_images}")

    def _defisheye(self, batch_images):
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

        undistorted_batch_images = []
        for img in batch_images:
            # undistort
            img_dst = cv2.remap(img, self.mapx, self.mapy, cv2.INTER_LINEAR)

            # crop the image
            x, y, w, h = self.roi
            img_dst = img_dst[y : y + h, x : x + w]

            undistorted_batch_images.append(img_dst)
        return undistorted_batch_images

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

    def _process_results(self, model_results: Results):
        for r in model_results:
            result = r.cpu()
            boxes = result.boxes.numpy()

            logger.debug(f"Target classes: {self.target_classes}")
            logger.debug(f"Boxes classes: {boxes.cls}")
            logger.debug(f"Boxes: {boxes}")
            target_idxs = np.where(np.in1d(boxes.cls, self.target_classes))[0]
            if len(target_idxs) == 0:  # Nothing to do!
                logger.debug("No container detected.")
                continue

            image = result.orig_img.copy()
            sensitive_idxs = np.where(np.in1d(boxes.cls, self.sensitive_classes))[0]

            # Blur sensitive data
            sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
            image = blurring_tools.blur_inside_boxes(image, sensitive_bounding_boxes)

            # Draw annotation boxes
            target_bounding_boxes = boxes[target_idxs].xyxy
            image = blurring_tools.draw_bounding_boxes(image, target_bounding_boxes)

            # Save image
            folder_path = os.path.join(
                self.detections_folder, os.path.basename(os.path.dirname(result.path))
            )
            logger.debug(f"Folder path: {folder_path}")
            pathlib.Path(folder_path).mkdir(parents=True, exist_ok=True)
            logger.debug("Created folder.")
            save_path = os.path.join(folder_path, pathlib.Path(result.path).name)
            logger.debug(f"Save path: {save_path}")
            cv2.imwrite(save_path, image)
            logger.debug("Saved image.")

            # Save annotation
            annotation_str = self._get_annotion_string_from_boxes(boxes[target_idxs])
            base_name = pathlib.Path(result.path).stem
            annotation_path = os.path.join(folder_path, f"{base_name}.txt")
            with open(annotation_path, "w") as f:
                f.write(annotation_str)
