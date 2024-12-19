from typing import List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt
from yolo_model_development_kit.inference_pipeline.source.output_image import (
    OutputImage,
)


class OOROutputImage(OutputImage):

    def __init__(self, image: npt.NDArray):
        super().__init__(image)
        self.shape = image.shape

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
        img_height, img_width, _ = self.image.shape  # type: ignore

        blurred_image = cv2.GaussianBlur(
            self.image, (blur_kernel_size, blur_kernel_size), 0  # type: ignore
        )

        for box in boxes:
            x_min, y_min, x_max, y_max = map(int, box)

            x_min = max(0, x_min - box_padding)
            y_min = max(0, y_min - box_padding)
            x_max = min(img_width, x_max + box_padding)
            y_max = min(img_height, y_max + box_padding)

            # logger.debug(f"Blurring outside: {(x_min, y_min)} -> {(x_max, y_max)}")
            blurred_image[y_min:y_max, x_min:x_max] = self.image[  # type: ignore
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
