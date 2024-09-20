import logging
import os
import secrets
from collections import defaultdict
from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from ultralytics.engine.results import Boxes, Results

logger = logging.getLogger("inference_pipeline")

# Predefined colors for 5 categories
DEFAULT_COLORS = {
    0: (0, 0, 255),  # Blue
    1: (0, 255, 0),  # Green
    2: (255, 0, 0),  # Red
    3: (255, 255, 0),  # Cyan
    4: (255, 0, 255),  # Magenta
}


class ModelResult:
    def __init__(
        self,
        model_result: Results,
        target_classes: List[int],
        sensitive_classes: List[int],
        target_classes_conf: float,
        sensitive_classes_conf: float,
        save_image: bool = False,
        save_labels: bool = True,
    ) -> None:
        self.result = model_result.cpu()
        self.image = self.result.orig_img.copy()
        self.boxes = self.result.boxes.numpy()
        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.target_classes_conf = target_classes_conf
        self.sensitive_classes_conf = sensitive_classes_conf
        self.save_image = save_image
        self.save_labels = save_labels

        # Initialize the category_colors dictionary with predefined colors and add random colors for new categories
        self.category_colors = defaultdict(
            lambda: (
                secrets.randbelow(256),
                secrets.randbelow(256),
                secrets.randbelow(256),
            ),
            DEFAULT_COLORS,
        )

    def process_detections_and_blur_sensitive_data(
        self,
        image_output_path: Union[str, os.PathLike],
        image_file_name: Union[str, os.PathLike],
        labels_output_path: Union[str, os.PathLike] = None,
    ) -> int:
        for summary_str in self._yolo_result_summary():
            logger.info(summary_str)

        target_idxs = np.where(
            np.in1d(self.boxes.cls, self.target_classes)
            & (self.boxes.conf >= self.target_classes_conf)
        )[0]
        if len(target_idxs) == 0:
            logger.debug("No container detected, not storing the image.")
            return 0

        sensitive_idxs = np.where(
            np.in1d(self.boxes.cls, self.sensitive_classes)
            & (self.boxes.conf >= self.sensitive_classes_conf)
        )[0]
        if len(sensitive_idxs) > 0:
            sensitive_bounding_boxes = self.boxes[sensitive_idxs].xyxy
            self.blur_inside_boxes(boxes=sensitive_bounding_boxes)

        target_bounding_boxes = self.boxes[target_idxs].xyxy
        target_categories = [int(box.cls) for box in self.boxes[target_idxs]]
        self.draw_bounding_boxes(
            boxes=target_bounding_boxes,
            categories=target_categories,
            colour_map=self.category_colors,
        )

        self._save_result(
            target_idxs, image_output_path, image_file_name, labels_output_path
        )

        return len(target_idxs)

    def _save_result(
        self,
        target_idxs: List[int],
        image_output_path: Union[str, os.PathLike],
        image_file_name: Union[str, os.PathLike],
        labels_output_path: Optional[Union[str, os.PathLike]] = None,
    ) -> None:
        if self.save_image:
            os.makedirs(image_output_path, exist_ok=True)
            image_full_path = os.path.join(image_output_path, image_file_name)
            cv2.imwrite(image_full_path, self.image)
            logger.debug(f"Image saved: {image_full_path}")

        if self.save_labels:
            if labels_output_path:
                os.makedirs(labels_output_path, exist_ok=True)
            else:
                labels_output_path = image_output_path

            img_name = os.path.splitext(os.path.basename(image_file_name))[0]
            labels_full_path = os.path.join(labels_output_path, f"{img_name}.txt")
            annotation_str = self._get_annotation_string_from_boxes(
                self.boxes[target_idxs]
            )
            with open(labels_full_path, "w") as f:
                f.write(annotation_str)
            logger.debug(f"Labels saved: {labels_full_path}")

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

    def _yolo_result_summary(self) -> List[str]:
        """Returns a readable summary of the results.

        Returns
        -------
        Dict
            Readable summary of objects detected and compute used.
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

    def yolo_annotation_to_bounds(
        self, yolo_annotation: str, img_shape: Tuple[int, int]
    ) -> Tuple[int, int, int, int]:
        """
        Convert YOLO annotation with normalized values to absolute bounds.

        Parameters
        ----------
        yolo_annotation : str
            YOLO annotation string in the format:
            "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
        img_shape : Tuple[int, int]
            Image dimensions as tuple (width, height)

        Returns
        -------
        tuple
            A tuple (x_min, y_min, x_max, y_max).
        """
        _, x_center_norm, y_center_norm, w_norm, h_norm = map(
            float, yolo_annotation.split()[0:5]
        )

        x_center_abs = x_center_norm * img_shape[0]
        y_center_abs = y_center_norm * img_shape[1]
        w_abs = w_norm * img_shape[0]
        h_abs = h_norm * img_shape[1]

        x_min = int(x_center_abs - (w_abs / 2))
        y_min = int(y_center_abs - (h_abs / 2))
        x_max = int(x_center_abs + (w_abs / 2))
        y_max = int(y_center_abs + (h_abs / 2))

        return x_min, y_min, x_max, y_max

    def blur_inside_boxes(
        self,
        boxes: Union[List[Tuple[float, float, float, float]], npt.NDArray[np.float_]],
        blur_kernel_size: int = 165,
        box_padding: int = 0,
    ) -> None:
        """
        Apply GaussianBlur with given kernel size to the area given by the bounding box(es).

        Parameters
        ----------
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) of the area(s) to blur, in the format (xmin, ymin, xmax, ymax).
        blur_kernel_size : int (default: 165)
            Kernel size (used for both width and height) for GaussianBlur.
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before applying the blur.

        Return
        ------
        image : numpy.ndarray
            The image blurred.
        """
        img_height, img_width, _ = self.image.shape

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            # logger.debug(f"Blurring inside: {(x_min, y_min)} -> {(x_max, y_max)}")
            area_to_blur = self.image[y_min:y_max, x_min:x_max]
            blurred = cv2.GaussianBlur(
                area_to_blur, (blur_kernel_size, blur_kernel_size), 0
            )
            self.image[y_min:y_max, x_min:x_max] = blurred

    def blur_outside_boxes(
        self,
        boxes: Union[List[Tuple[float, float, float, float]], npt.NDArray[np.float_]],
        blur_kernel_size: int = 165,
        box_padding: int = 0,
    ) -> None:
        """
        Apply GaussianBlur with given kernel size to the area outside the given bounding box(es).

        Parameters
        ----------
        image : numpy.ndarray
            The image to blur.
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) outside which to blur, in the format (xmin, ymin, xmax, ymax).
        blur_kernel_size : int (default: 165)
            Kernel size (used for both width and height) for GaussianBlur.
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before applying the blur.
        """
        img_height, img_width, _ = self.image.shape

        blurred_image = cv2.GaussianBlur(
            self.image, (blur_kernel_size, blur_kernel_size), 0
        )

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            # logger.debug(f"Blurring outside: {(x_min, y_min)} -> {(x_max, y_max)}")
            blurred_image[y_min:y_max, x_min:x_max] = self.image[
                y_min:y_max, x_min:x_max
            ]

        self.image = blurred_image

    def crop_outside_boxes(
        self,
        boxes: Union[List[Tuple[float, float, float, float]], npt.NDArray[np.float_]],
        box_padding: int = 0,
        fill_bg: bool = False,
    ) -> List[npt.NDArray[np.int_]]:
        """
        Crop image to the area(s) given by the yolo annotation box(es).

        When multiple bounding boxes are provided and fill_bg is False, multiple cropped images will be returned.
        When multiple bounding boxes are provided and fill_bg is True, a single image will be returned.

        Parameters
        ----------
        image : numpy.ndarray
            The image to blur.
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) of the area(s) to crop, in the format (xmin, ymin, xmax, ymax).
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before cropping.
        fill_bg : bool (default: False)
            Instead of cropping, fill the background with white.

        Returns
        -------
        List[numpy.ndarray]
            The cropped image(s)
        """
        img_height, img_width, _ = self.image.shape

        cropped_images = []

        if fill_bg:
            cropped_images.append(np.ones_like(self.image) * 255)

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            # logger.debug(f"Cropping: {(x_min, y_min)} -> {(x_max, y_max)}")
            if not fill_bg:
                cropped_images.append(self.image[y_min:y_max, x_min:x_max].copy())
            else:
                cropped_images[0][y_min:y_max, x_min:x_max] = self.image[
                    y_min:y_max, x_min:x_max
                ].copy()

        return cropped_images

    def draw_bounding_boxes(
        self,
        boxes: Union[List[Tuple[float, float, float, float]], npt.NDArray[np.float_]],
        categories: Optional[List[int]] = None,
        colour_map: Dict[int, Tuple[int, int, int]] = DEFAULT_COLORS,
        box_padding: int = 0,
        line_thickness: int = 3,
        tracking_ids: Optional[List[int]] = None,
        font_scale: float = 0.7,
        font_thickness: int = 2,
    ) -> None:
        """
        Draw the given bounding box(es).

        Parameters
        ----------
        image : numpy.ndarray
            The image to draw on.
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) to draw, in the format (xmin, ymin, xmax, ymax).
        categories : Optional[List[int]] (default: None)
            The category of each bounding box. If not provided, colour is set to "red".
        colour_map : Dict[int, Tuple[int, int, int]]
            Dictionary of colours for each category, in the format {category: (255, 255, 255)}.
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before drawing.
        line_thickness : int (default: 3)
            Line thickness for the bounding box.
        tracking_ids : Optional[List[int]] (default: None)
            Optional: list of tracking IDs for each bounding box. If not provided, no tracking IDs are drawn.
        font_scale : float (default: 0.7)
            Font scale for the text.
        font_thickness : int (default: 2)
            Thickness of the text.
        """
        img_height, img_width, _ = self.image.shape

        if categories:
            colours = [colour_map[category] for category in categories]
        else:
            colours = [(255, 0, 0)] * len(boxes)

        for i, (box, colour) in enumerate(zip(boxes, colours)):

            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            # logger.debug(
            #     f"Drawing: {(x_min, y_min)} -> {(x_max, y_max)} in colour {colour}"
            # )

            self.image = cv2.rectangle(
                self.image,
                (x_min, y_min),
                (x_max, y_max),
                colour,
                thickness=line_thickness,
            )

            if tracking_ids and tracking_ids[i] != -1:
                text = f"ID: {tracking_ids[i]}"
                (text_width, text_height), baseline = cv2.getTextSize(
                    text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
                )
                cv2.rectangle(
                    self.image,
                    (x_min, y_min - text_height - baseline),
                    (x_min + text_width, y_min),
                    colour,
                    thickness=cv2.FILLED,
                )
                cv2.putText(
                    self.image,
                    text,
                    (x_min, y_min - baseline),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale,
                    (255, 255, 255),
                    font_thickness,
                    lineType=cv2.LINE_AA,
                )
