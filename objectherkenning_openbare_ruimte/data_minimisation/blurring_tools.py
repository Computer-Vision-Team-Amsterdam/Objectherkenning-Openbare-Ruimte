from typing import Tuple

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


def blur_inside_yolo_box(
    image: npt.NDArray[np.int_],
    yolo_annotation: str,
    blur_kernel_size: int = 165,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
    """
    Apply GaussianBlur with given kernel size to the area given by the yolo annotation box.

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
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

    x_min, y_min, x_max, y_max = yolo_annotation_to_bounds(
        yolo_annotation, img_height, img_width
    )
    x_min = max(0, x_min - box_padding)
    y_min = max(0, y_min - box_padding)
    x_max = min(img_width, x_max + box_padding)
    y_max = min(img_height, y_max + box_padding)
    print(f"Blurring inside: {(x_min, y_min)} -> {(x_max, y_max)}")
    area_to_blur = image[y_min:y_max, x_min:x_max]
    blurred = cv2.GaussianBlur(area_to_blur, (blur_kernel_size, blur_kernel_size), 0)
    image[y_min:y_max, x_min:x_max] = blurred
    return image


def blur_outside_yolo_box(
    image: npt.NDArray[np.int_],
    yolo_annotation: str,
    blur_kernel_size: int = 165,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
    """
    Apply GaussianBlur with given kernel size to the area given by the yolo annotation box.

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
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

    x_min, y_min, x_max, y_max = yolo_annotation_to_bounds(
        yolo_annotation, img_height, img_width
    )
    x_min = max(0, x_min - box_padding)
    y_min = max(0, y_min - box_padding)
    x_max = min(img_width, x_max + box_padding)
    y_max = min(img_height, y_max + box_padding)
    print(f"Blurring outside: {(x_min, y_min)} -> {(x_max, y_max)}")
    area_to_keep = image[y_min:y_max, x_min:x_max]
    blurred_image = cv2.GaussianBlur(image, (blur_kernel_size, blur_kernel_size), 0)
    blurred_image[y_min:y_max, x_min:x_max] = area_to_keep
    return blurred_image


def crop_outside_yolo_box(
    image: npt.NDArray[np.int_],
    yolo_annotation: str,
    box_padding: int = 0,
    fill_bg: bool = False,
) -> npt.NDArray[np.int_]:
    """
    Crop image to the area given by the yolo annotation box.

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
    box_padding : int (default: 0)
        Optional: increase box by this amount of pixels before cropping.
    fill_bg : bool (default: False)
        Instead of cropping, fill the backrgound with white.

    Returns
    -------
    numpy.ndarray
        The cropped image
    """
    image = image.copy()
    img_height, img_width, _ = image.shape

    x_min, y_min, x_max, y_max = yolo_annotation_to_bounds(
        yolo_annotation, img_height, img_width
    )
    x_min = max(0, x_min - box_padding)
    y_min = max(0, y_min - box_padding)
    x_max = min(img_width, x_max + box_padding)
    y_max = min(img_height, y_max + box_padding)
    print(f"Cropping: {(x_min, y_min)} -> {(x_max, y_max)}")
    if not fill_bg:
        return image[y_min:y_max, x_min:x_max]
    else:
        area_to_keep = image[y_min:y_max, x_min:x_max].copy()
        image[:, :, :] = 255
        image[y_min:y_max, x_min:x_max] = area_to_keep
        return image


def draw_yolo_box(
    image: npt.NDArray[np.int_],
    yolo_annotation: str,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
    """
    Draw the given yolo annotation box.

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    yolo_annotation : str
        YOLO annotation string in the format:
        "<class_id> <x_center_norm> <y_center_norm> <w_norm> <h_norm>".
    box_padding : int (default: 0)
        Optional: increase box by this amount of pixels before drawing.

    Returns
    -------
    numpy.ndarray
        The image with drawn bounding box.
    """
    img_height, img_width, _ = image.shape

    x_min, y_min, x_max, y_max = yolo_annotation_to_bounds(
        yolo_annotation, img_height, img_width
    )
    x_min = max(0, x_min - box_padding)
    y_min = max(0, y_min - box_padding)
    x_max = min(img_width, x_max + box_padding)
    y_max = min(img_height, y_max + box_padding)
    print(f"Drawing: {(x_min, y_min)} -> {(x_max, y_max)}")
    image = cv2.rectangle(
        image, (x_min, y_min), (x_max, y_max), (0, 0, 255), thickness=2
    )
    return image
