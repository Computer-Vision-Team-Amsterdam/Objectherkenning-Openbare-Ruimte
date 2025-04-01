import cv2
import numpy.typing as npt


class OutputImage:

    def __init__(self, image: npt.NDArray):
        """
        Initialize with the original image.
        """
        self.image = image

    def get_image(self) -> npt.NDArray:
        """
        Returns the image as Numpy array.
        """
        return self.image

    def draw_bounding_boxes(
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
