import base64
from typing import Tuple, Union

import cv2
import numpy.typing as npt


class OutputImage:

    def __init__(self, image_path: str):
        """
        Initialize with the original image.
        """
        self.image = cv2.imread(image_path)

    def get_image(self) -> npt.NDArray:
        """
        Returns the image as Numpy array.
        """
        return self.image

    def b64encode(
        self, shrink_to_size: Union[int, Tuple[int, int]] = (1280, 720)
    ) -> str:
        self.shrink(shrink_to_size)

        # Convert image to JPG
        _, buffer = cv2.imencode(".jpg", self.image)

        # Convert to base64 encoding
        b64str = base64.b64encode(buffer).decode("utf-8")

        return b64str

    def shrink(self, size: Union[int, Tuple[int, int]] = (1280, 720)):
        """
        Shrink the image to the desired size.

        Parameters
        ----------
        size: Union[int, Tuple[int, int]]
            New size either as int max_dim, or as tuple (width, height)
        """
        img_height, img_width, _ = self.image.shape
        if isinstance(size, int):
            if (img_height > img_width) and (img_height > size):
                new_width = int(img_width * (size / img_height))
                size = (new_width, size)
            elif img_width > size:
                new_height = int(img_height * (size / img_width))
                size = (size, new_height)
            else:
                size = (img_width, img_height)

        if isinstance(size, tuple):
            if (img_height != size[1]) or (img_width != size[0]):
                self.image = cv2.resize(self.image, size)
        else:
            raise ValueError(f"Illegal value for arg size: {type(size)}")

    def draw_bounding_box(
        self,
        x_center_norm: float,
        y_center_norm: float,
        width_norm: float,
        height_norm: float,
        line_thickness: int = 3,
    ) -> None:
        """
        Draw bounding boxes on the image.
        """
        img_height, img_width, _ = self.image.shape

        # Convert normalized values to pixel coordinates
        x_center = int(x_center_norm * img_width)
        y_center = int(y_center_norm * img_height)
        box_width = int(width_norm * img_width)
        box_height = int(height_norm * img_height)

        # Compute the top-left and bottom-right points
        x_min = max(0, x_center - box_width // 2)
        y_min = max(0, y_center - box_height // 2)
        x_max = min(img_width, x_center + box_width // 2)
        y_max = min(img_height, y_center + box_height // 2)

        if x_min < x_max and y_min < y_max:
            cv2.rectangle(
                self.image,
                (x_min, y_min),
                (x_max, y_max),
                color=(0, 0, 255),
                thickness=line_thickness,
            )
