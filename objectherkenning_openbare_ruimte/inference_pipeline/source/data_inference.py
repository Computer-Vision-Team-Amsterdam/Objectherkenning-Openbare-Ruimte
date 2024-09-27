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
        save_images: bool = False,
        save_labels: bool = True,
        save_all_images: bool = False,
        save_images_subfolder: Optional[str] = None,
        save_labels_subfolder: Optional[str] = None,
        batch_size: int = 1,
    ) -> None:
        """
        This class runs inference on images using a pre-trained YOLOv8 model to
        detect a set of target classes. Optionally, sensitive classes are
        blurred and output images are stored.

        Input images will be de-fisheyed and resized if needed. When one or more
        objects belonging to one of the target classes are detected in an image,
        the bounding boxes for those detections are stored in a .txt file with
        the same name as the image. If save_images or save_all_images is set,
        the output will be saved as an image with the original file name, with
        sensitive classes blurred, and bounding boxes of target classes drawn.

        The difference between save_image and save_all_images is that the former
        will only save images when an object belonging to one of the
        target_classes is detected in the image.

        Parameters
        ----------
        images_folder: str
            Location of images to run inference on. If the location contains
            sub-folders, this folder structure will be preserved in the output.
        output_folder: str
            Location where output (annotation labels and possible images) will
            be stored.
        model_path: str
            Location of the pre-trained YOLOv8 model.
        inference_params: Dict
            Inference parameters for the YOLOv8 model.
        target_classes: List
            List of target classes for which bounding boxes will be predicted.
        sensitive_classes: List
            List of sensitive classes which will be blurred in output images.
        target_classes_conf: Optional[float] = None
            Optional: confidence threshold for target classes. Only detections
            above this threshold will be considered. If omitted,
            inference_param["conf"] will be used.
        sensitive_classes_conf: Optional[float] = None
            Optional: confidence threshold for sensitive classes. Only
            detections above this threshold will be considered. If omitted,
            inference_param["conf"] will be used.
        output_image_size: Optional[Tuple[int, int]] = None
            Optional: output images will be resized to these (width, height)
            dimensions if set.
        defisheye_flag: bool = False
            Whether or not to apply distortion correction to the input images.
        defisheye_params: Dict = {}
            If defisheye_flag is True, these distortion correction parameters
            will be used. Contains "camera_matrix", "distortion_params", and
            "input_image_size" (size of images used to compute these
            parameters).
        save_images: bool = False
            Whether or not to save the output images.
        save_labels: bool = True
            Whether or not to save the annotation labels.
        save_all_images: bool = False
            Whether to save all processed images (TRue) or only those containing
            objects belonging to one of the target classes (False).
        save_images_subfolder: Optional[str] = None
            Optional: sub-folder in which to store output images.
        save_labels_subfolder: Optional[str] = None
            Optional: sub-folder in which to store annotation labels.
        batch_size: int = 1
            Batch size for inference.
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

        self.save_detections = save_images
        self.save_labels = save_labels
        self.save_all_images = save_all_images
        self.detections_subfolder = (
            save_images_subfolder if save_images_subfolder else ""
        )
        self.labels_subfolder = save_labels_subfolder if save_labels_subfolder else ""

        self.batch_size = batch_size

    def run_pipeline(self) -> None:
        """
        Runs the inference pipeline:
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
        """
        Process all images in all sub-folders in batches of size batch_size.

        Parameters
        ----------
        folders_and_frames: Dict[str, List[str]]
            Dictionary mapping folder names to the images they contain as
            `{"folder_name": List[image_names]}`
        """
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
        """
        Process the YOLOv8 inference Results objects:
        - save output image
        - save annotation labels

        Parameters
        ----------
        model_results: List[Results]
            List of YOLOv8 inference Results objects.
        image_paths: List[str]
            List of input image paths corresponding to the Results.
        """
        for result, image_path in zip(model_results, image_paths):
            model_result = ModelResult(
                model_result=result,
                target_classes=self.target_classes,
                sensitive_classes=self.sensitive_classes,
                target_classes_conf=self.target_classes_conf,
                sensitive_classes_conf=self.sensitive_classes_conf,
                save_image=self.save_detections,
                save_labels=self.save_labels,
                save_all_images=self.save_all_images,
            )

            # Get the relative path of the image w.r.t. the input folder. This
            # is used to preserve the folder structure in the output.
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
                output_folder=image_output_path,
                image_file_name=image_file_name,
                labels_output_folder=labels_output_path,
            )

    @staticmethod
    def _find_image_paths_and_group_by_folder(root_folder: str) -> Dict[str, List[str]]:
        """
        Find all image files in the root_folder, group them by sub-folder, and
        return these as a dictionary mapping.

        Parameters
        ----------
        root_folder: str
            The root folder.

        Returns
        -------
        A dictionary mapping sub-folders to images contained in them:
        `{"folder_name": ["img_1.jpg", "img_2.jpg", ...]}`
        """
        folders_and_frames: Dict[str, List[str]] = {}
        for foldername, _, filenames in os.walk(root_folder):
            for filename in filenames:
                if any(filename.endswith(ext) for ext in IMG_FORMATS):
                    if foldername not in folders_and_frames:
                        folders_and_frames[foldername] = [filename]
                    else:
                        folders_and_frames[foldername].append(filename)
        return folders_and_frames
