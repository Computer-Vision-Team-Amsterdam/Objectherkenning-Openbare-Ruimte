from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt


def yolo_annotation_to_bounds(
    yolo_annotation: str, img_height: int, img_width: int
) -> Tuple[int, int, int, int]:
    """
    Convert YOLO annotation with normalized values to absolute bounds.

    Parameters
    ----------
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
    img_height : int
        Height of the image.
    img_width : int
        Width of the image.

    Returns
    -------
    tuple
        A tuple (x_min, y_min, x_max, y_max).
    """
    _, x_center_norm, y_center_norm, w_norm, h_norm = map(
        float, yolo_annotation.split()
    )

    x_center_abs = x_center_norm * img_width
    y_center_abs = y_center_norm * img_height
    w_abs = w_norm * img_width
    h_abs = h_norm * img_height

    x_min = int(x_center_abs - (w_abs / 2))
    y_min = int(y_center_abs - (h_abs / 2))
    x_max = int(x_center_abs + (w_abs / 2))
    y_max = int(y_center_abs + (h_abs / 2))

    return x_min, y_min, x_max, y_max


def blur_inside_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[float, float, float, float]],
    blur_kernel_size: int = 165,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
    """
    Apply GaussianBlur with given kernel size to the area given by the bounding box(es).

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    boxes : List[Tuple[float, float, float, float]]
        Bounding box(es) of the area(s) to blur, in the format (xmin, ymin, xmax, ymax).
    blur_kernel_size : int (default: 165)
        Kernel size (used for both width and height) for GaussianBlur.
    box_padding : int (default: 0)
        Optional: increase box by this amount of pixels before applying the blur.

    Returns
    -------
    numpy.ndarray
        The blurred image
    """
    image = image.copy()
    img_height, img_width, _ = image.shape

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Blurring inside: {(x_min, y_min)} -> {(x_max, y_max)}")
        area_to_blur = image[y_min:y_max, x_min:x_max]
        blurred = cv2.GaussianBlur(
            area_to_blur, (blur_kernel_size, blur_kernel_size), 0
        )
        image[y_min:y_max, x_min:x_max] = blurred

    return image


def blur_outside_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[float, float, float, float]],
    blur_kernel_size: int = 165,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
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

    Returns
    -------
    numpy.ndarray
        The blurred image
    """
    img_height, img_width, _ = image.shape

    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Blurring outside: {(x_min, y_min)} -> {(x_max, y_max)}")
        blurred_image[y_min:y_max, x_min:x_max] = image[y_min:y_max, x_min:x_max]

    return blurred_image


def crop_outside_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[float, float, float, float]],
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
        Instead of cropping, fill the backrgound with white.

    Returns
    -------
    List[numpy.ndarray]
        The cropped image(s)
    """
    img_height, img_width, _ = image.shape

    cropped_images = []

    if fill_bg:
        cropped_images.append(np.ones_like(image) * 255)

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Cropping: {(x_min, y_min)} -> {(x_max, y_max)}")
        if not fill_bg:
            cropped_images.append(image[y_min:y_max, x_min:x_max].copy())
        else:
            cropped_images[0][y_min:y_max, x_min:x_max] = image[
                y_min:y_max, x_min:x_max
            ].copy()

    return cropped_images


def draw_bounding_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[float, float, float, float]],
    colours: List[Tuple[int, int, int]] = [(0, 0, 255)],
    box_padding: int = 0,
    line_thickness: int = 2,
) -> npt.NDArray[np.int_]:
    """
    Draw the given bounding box(es).

    Parameters
    ----------
    image : numpy.ndarray
        The image to draw on.
    boxes : List[Tuple[float, float, float, float]]
        Bounding box(es) to draw, in the format (xmin, ymin, xmax, ymax).
    colours : List[Tuple[int, int, int]] (default: [(0, 0, 255)])
        Optional: list of colours for each bounding box, in the format (255, 255, 255)
    box_padding : int (default: 0)
        Optional: increase box by this amount of pixels before drawing.
    line_thickness : int (default: 2)
        Line thickness for the bounding box.

    Returns
    -------
    numpy.ndarray
        The image with drawn bounding box.
    """
    img_height, img_width, _ = image.shape

    if len(colours) < len(boxes):
        difference = len(boxes) - len(colours)
        colours.extend([colours[-1]] * difference)

    for colour, box in zip(colours, boxes):
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Drawing: {(x_min, y_min)} -> {(x_max, y_max)} in colour {colour}")
        image = cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), colour, thickness=line_thickness
        )
    return image
