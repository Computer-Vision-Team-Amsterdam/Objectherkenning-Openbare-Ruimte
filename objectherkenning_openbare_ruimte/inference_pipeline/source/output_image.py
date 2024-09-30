from typing import Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt


class OutputImage:
    # Predefined colors for 5 categories
    DEFAULT_COLORS: Dict[int, Tuple[int, int, int]] = {
        0: (0, 0, 255),  # Blue
        1: (0, 255, 0),  # Green
        2: (255, 0, 0),  # Red
        3: (255, 255, 0),  # Cyan
        4: (255, 0, 255),  # Magenta
    }

    def __init__(self, image: npt.NDArray):
        """
        This class is used to blur and annotate an output image based on model predictions.

        Parameters
        ----------
        image: npt.NDArray
            The original image.
        """
        self.image = image

    def get_image(self) -> npt.NDArray:
        """Returns the image as Numpy array."""
        return self.image

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

        When multiple bounding boxes are provided and fill_bg is False, multiple
        cropped images will be returned. When multiple bounding boxes are
        provided and fill_bg is True, a single image will be returned.

        *NOTE*: in contrast to the other methods of the class,
        crop_outside_boxes does not modify the image in place but instead
        returns one or more cropped images. When fill_bg is set to True, the
        image will additionally be modified in place.

        Parameters
        ----------
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) of the area(s) to crop, in the format (xmin, ymin,
            xmax, ymax).
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

        if fill_bg:
            self.image = cropped_images[0]

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
        boxes : List[Tuple[float, float, float, float]]
            Bounding box(es) to draw, in the format (xmin, ymin, xmax, ymax).
        categories : Optional[List[int]] (default: None)
            Optional: the category of each bounding box. If not provided, colour
            is set to "red".
        colour_map : Dict[int, Tuple[int, int, int]]
            Dictionary of colours for each category, in the format `{category:
            (255, 255, 255)}`.
        box_padding : int (default: 0)
            Optional: increase box by this amount of pixels before drawing.
        line_thickness : int (default: 3)
            Line thickness for the bounding box.
        tracking_ids : Optional[List[int]] (default: None)
            Optional: list of tracking IDs for each bounding box. If not
            provided, no tracking IDs are drawn.
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
