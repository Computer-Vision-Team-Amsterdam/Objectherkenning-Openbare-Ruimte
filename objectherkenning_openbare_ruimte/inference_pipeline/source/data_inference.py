import logging
import os
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import numpy.typing as npt
import torch
from cvtoolkit.helpers.file_helpers import IMG_FORMATS
from ultralytics import YOLO
from ultralytics.engine.results import Results

from objectherkenning_openbare_ruimte.inference_pipeline.source.input_image import (
    InputImage,
)
from objectherkenning_openbare_ruimte.inference_pipeline.source.model_result import (
    ModelResult,
)

logger = logging.getLogger("inference_pipeline")


class DataInference:
    def __init__(
        self,
        images_folder: str,
        output_folder: str,
        model_path: str,
        inference_params: Dict,
        target_classes: List,
        sensitive_classes: List,
        target_classes_conf: Optional[float] = None,
        sensitive_classes_conf: Optional[float] = None,
        output_image_size: Optional[Tuple[int, int]] = None,
        defisheye_flag: bool = False,
        defisheye_params: Dict = {},
        save_detections: bool = False,
        save_labels: bool = True,
        detections_subfolder: str = "",
        labels_subfolder: str = "",
        batch_size: int = 1,
    ) -> None:
        """
        Object that find containers in the images using a pre-trained YOLO model and blurs sensitive data.

        Parameters
        ----------
        images_folder
            Folder containing images to run detection on.
        """
        self.images_folder = images_folder
        self.output_folder = output_folder
        self.inference_params = inference_params
        logger.debug(f"Inference_params: {self.inference_params}")
        logger.debug(f"YOLO model: {model_path}")
        logger.debug(f"Output folder: {self.output_folder}")

        self.model = YOLO(model=model_path, task="detect")

        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.target_classes_conf = (
            target_classes_conf if target_classes_conf else inference_params["conf"]
        )
        self.sensitive_classes_conf = (
            sensitive_classes_conf
            if sensitive_classes_conf
            else inference_params["conf"]
        )

        logger.debug(
            f"Using confidence thresholds: target_classes: {self.target_classes_conf}, "
            f"sensitive_classes: {self.sensitive_classes_conf}"
        )

        self.output_image_size = output_image_size
        self.defisheye_flag = defisheye_flag
        self.defisheye_params = defisheye_params
        self.mapx = self.mapy = None

        self.save_detections = save_detections
        self.save_labels = save_labels
        self.detections_subfolder = detections_subfolder
        self.labels_subfolder = labels_subfolder

        self.batch_size = batch_size

    def run_pipeline(self) -> None:
        """
        Runs the detection pipeline:
            - find the images to detect;
            - detects everything;
            - stores labels if required;
            - stores images if required, with
              - sensitive classes blurred;
              - target classes bounding boxes drawn;
        """
        logger.info(f"Running detection pipeline on {self.images_folder}..")
        folders_and_frames = self._find_image_paths_and_group_by_folder(
            root_folder=self.images_folder
        )
        logger.info(
            f"Total number of images: {sum(len(frames) for frames in folders_and_frames.values())}"
        )
        self._process_batches(folders_and_frames=folders_and_frames)

    def _load_and_defisheye_image(
        self, image_path: Union[os.PathLike, str]
    ) -> npt.NDArray[np.float_]:
        image = InputImage(image_full_path=str(image_path))
        if self.output_image_size:
            image.resize(output_image_size=self.output_image_size)
        if self.defisheye_flag:
            image.defisheye(defisheye_params=self.defisheye_params)
        return image.image

    def _process_batches(self, folders_and_frames: Dict[str, List[str]]) -> None:
        for folder_name, images in folders_and_frames.items():
            logger.debug(
                f"Running inference on folder: {os.path.relpath(folder_name, self.images_folder)}"
            )
            image_paths = [os.path.join(folder_name, image) for image in images]
            logger.debug(f"Number of images to detect: {len(image_paths)}")
            processed_images = 0
            for i in range(0, len(image_paths), self.batch_size):
                batch_images = [
                    self._load_and_defisheye_image(image_path)
                    for image_path in image_paths[i : i + self.batch_size]
                ]
                self.inference_params["source"] = batch_images
                self.inference_params["name"] = folder_name
                batch_results = self.model(**self.inference_params)
                torch.cuda.empty_cache()  # Clear unused memory
                self._process_detections_and_blur_image(
                    model_results=batch_results,
                    image_paths=image_paths[i : i + self.batch_size],
                )
                processed_images += len(batch_images)

            logger.debug(f"Number of images processed: {processed_images}")

    def _process_detections_and_blur_image(
        self, model_results: List[Results], image_paths: List[str]
    ) -> None:
        for result, image_path in zip(model_results, image_paths):
            model_result = ModelResult(
                model_result=result,
                target_classes=self.target_classes,
                sensitive_classes=self.sensitive_classes,
                target_classes_conf=self.target_classes_conf,
                sensitive_classes_conf=self.sensitive_classes_conf,
                save_image=self.save_detections,
                save_labels=self.save_labels,
            )

            base_folder = os.path.dirname(
                os.path.relpath(image_path, self.images_folder)
            )
            output_base_path = os.path.join(self.output_folder, base_folder)
            image_output_path = os.path.join(
                output_base_path, self.detections_subfolder
            )
            labels_output_path = os.path.join(output_base_path, self.labels_subfolder)
            image_file_name = os.path.basename(image_path)

            logger.debug(f"Processing and blurring image: {image_file_name}")

            model_result.process_detections_and_blur_sensitive_data(
                image_output_path=image_output_path,
                image_file_name=image_file_name,
                labels_output_path=labels_output_path,
            )

    @staticmethod
    def _find_image_paths_and_group_by_folder(root_folder: str) -> Dict[str, List[str]]:
        folders_and_frames: Dict[str, List[str]] = {}
        for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    if foldername not in folders_and_frames:
                        folders_and_frames[foldername] = [filename]
                    else:
                        folders_and_frames[foldername].append(filename)
        return folders_and_frames
