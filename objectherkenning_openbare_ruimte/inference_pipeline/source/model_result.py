import logging
import os
import secrets
from collections import defaultdict
from typing import List, Optional, Union

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.engine.results import Boxes, Results

from objectherkenning_openbare_ruimte.inference_pipeline.source.output_image import (
    OutputImage,
)

logger = logging.getLogger("inference_pipeline")


class ModelResult:
    def __init__(
        self,
        model_result: Results,
        target_classes: List[int],
        sensitive_classes: List[int],
        target_classes_conf: Optional[float] = None,
        sensitive_classes_conf: Optional[float] = None,
        save_image: bool = False,
        save_labels: bool = True,
        save_all_images: bool = False,
    ) -> None:
        """
        This class is used to process a Results object from YOLO inference.

        When one or more objects belonging to one of the target classes are
        detected in the image, the bounding boxes for those detections are
        stored in a .txt file with the same name as the image. If save_images or
        save_all_images is set, the output will be saved as an image with the
        original file name, with sensitive classes blurred, and bounding boxes
        of target classes drawn.

        The difference between save_image and save_all_images is that the former
        will only save images when an object belonging to one of the
        target_classes is detected in the image.

        Parameters
        ----------
        model_result: Results
            YOLO inference Results object for one image.
        target_classes: List
            List of target classes for which bounding boxes will be predicted.
        sensitive_classes: List
            List of sensitive classes which will be blurred in output image.
        target_classes_conf: Optional[float] = None
            Optional: confidence threshold for target classes. Only detections
            above this threshold will be considered. If omitted, all annotations
            will be used.
        sensitive_classes_conf: Optional[float] = None
            Optional: confidence threshold for sensitive classes. Only
            detections above this threshold will be considered. If omitted, all
            annotations will be used.
        save_image: bool = False
            Whether or not to save the output image.
        save_labels: bool = True
            Whether or not to save the annotation labels.
        save_all_images: bool = False
            Whether to save all processed images (TRue) or only those containing
            objects belonging to one of the target classes (False).
        """
        self.result = model_result.cpu()
        self.output_image = OutputImage(self.result.orig_img.copy())
        self.boxes = self.result.boxes.numpy()
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.target_classes_conf = target_classes_conf if target_classes_conf else 0.0
        self.sensitive_classes_conf = (
            sensitive_classes_conf if sensitive_classes_conf else 0.0
        )
        self.save_image = save_image
        self.save_labels = save_labels
        self.save_all_images = save_all_images

        # Initialize the category_colors dictionary with predefined colors and add random colors for new categories
        self.category_colors = defaultdict(
            lambda: (
                secrets.randbelow(256),
                secrets.randbelow(256),
                secrets.randbelow(256),
            ),
            OutputImage.DEFAULT_COLORS,
        )

    def process_detections_and_blur_sensitive_data(
        self,
        output_folder: Union[str, os.PathLike],
        image_file_name: Union[str, os.PathLike],
        labels_output_folder: Optional[Union[str, os.PathLike]] = None,
    ) -> int:
        """
        Process the detections present in the Results:

        - Save annotations in a.txt file
        - Save output image:
            - With sensitive classes blurred
            - With annotation boxes drawn for target classes.

        Parameters
        ----------
        output_folder: Union[str, os.PathLike]
            Folder where output will be saved.
        image_file_name: Union[str, os.PathLike]
            Filename for the output image.
        labels_output_folder: Optional[Union[str, os.PathLike]] = None
            Optional: provide a different folder where annotation labels will be
            stored. If omitted, the output_folder will be used.

        Returns
        -------
        The number of target class annotations saved.
        """
        for summary_str in self._yolo_result_summary():
            logger.info(summary_str)

        if not labels_output_folder:
            labels_output_folder = output_folder

        target_idxs = np.where(
            np.in1d(self.boxes.cls, self.target_classes)
            & (self.boxes.conf >= self.target_classes_conf)
        )[0]
        if len(target_idxs) == 0 and not self.save_all_images:
            logger.debug("No container detected, not storing the image.")
            return 0

        sensitive_idxs = np.where(
            np.in1d(self.boxes.cls, self.sensitive_classes)
            & (self.boxes.conf >= self.sensitive_classes_conf)
        )[0]
        if len(sensitive_idxs) > 0:
            sensitive_bounding_boxes = self.boxes[sensitive_idxs].xyxy
            self.output_image.blur_inside_boxes(boxes=sensitive_bounding_boxes)

        if len(target_idxs) > 0:
            target_bounding_boxes = self.boxes[target_idxs].xyxy
            target_categories = [int(box.cls) for box in self.boxes[target_idxs]]
            self.output_image.draw_bounding_boxes(
                boxes=target_bounding_boxes,
                categories=target_categories,
                colour_map=self.category_colors,
            )

        if self.save_image or self.save_all_images:
            self._save_image(output_folder, image_file_name)
        if self.save_labels and len(target_idxs) > 0:
            annotation_str = self._get_annotation_string_from_boxes(
                self.boxes[target_idxs]
            )
            self._save_labels(annotation_str, output_folder, image_file_name)

        return len(target_idxs)

    def get_image(self) -> npt.NDArray:
        """Returns the image."""
        return self.output_image.get_image()

    def _save_image(
        self,
        output_folder: Union[str, os.PathLike],
        image_file_name: Union[str, os.PathLike],
    ) -> None:
        """Save the image."""
        os.makedirs(output_folder, exist_ok=True)
        image_full_path = os.path.join(output_folder, image_file_name)
        cv2.imwrite(image_full_path, self.output_image.get_image())
        logger.debug(f"Image saved: {image_full_path}")

    def _save_labels(
        self,
        annotation_string: str,
        output_folder: Union[str, os.PathLike],
        image_file_name: Union[str, os.PathLike],
    ) -> None:
        """Save the annotation labels."""
        os.makedirs(output_folder, exist_ok=True)
        img_name = os.path.splitext(os.path.basename(image_file_name))[0]
        labels_full_path = os.path.join(output_folder, f"{img_name}.txt")
        with open(labels_full_path, "w") as f:
            f.write(annotation_string)
        logger.debug(f"Labels saved: {labels_full_path}")

    @staticmethod
    def _get_annotation_string_from_boxes(boxes: Boxes) -> str:
        """Generates and returns a YOLO-style string representation of the
        annotation bounding boxes."""
        boxes = boxes.cpu()
        annotation_lines = []

        for box in boxes:
            cls = int(box.cls.squeeze())
            conf = float(box.conf.squeeze())
            tracking_id = int(box.id.squeeze()) if box.is_track else -1
            yolo_box_str = " ".join([f"{x:.6f}" for x in box.xywhn.squeeze()])
            annotation_lines.append(f"{cls} {yolo_box_str} {conf:.6f} {tracking_id}")

        return "\n".join(annotation_lines)

    def _yolo_result_summary(self) -> List[str]:
        """
        Returns a readable summary of the results.

        Returns
        -------
        Readable summary of objects detected and compute used as two separate strings.
        """
        obj_classes, obj_counts = np.unique(self.result.boxes.cls, return_counts=True)
        obj_str = "Detected: {"
        for obj_cls, obj_count in zip(obj_classes, obj_counts):
            obj_str = obj_str + f"{self.result.names[obj_cls]}: {obj_count}, "
        if len(obj_classes):
            obj_str = obj_str[0:-2]
        obj_str = obj_str + "}"

        speed_str = "Compute: {"
        for key, value in self.result.speed.items():
            speed_str = speed_str + f"{key}: {value:.2f}ms, "
        speed_str = speed_str[0:-2] + "}"

        return [obj_str, speed_str]
