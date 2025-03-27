import cv2
import numpy.typing as npt


class OutputImage:

    def __init__(self, image: npt.NDArray):
        """
        Initialize with the original image.
        """
        self.image = image
        self.color_mapping = {
            2: (255, 0, 0),  # Blue
            3: (0, 255, 0),  # Green
            4: (0, 0, 255),  # Red
        }

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
        object_class: int,
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

        cv2.rectangle(
            self.image,
            (x_min, y_min),
            (x_max, y_max),
            color=self.color_mapping.get(object_class, (255, 255, 255)),
            thickness=line_thickness,
        )
