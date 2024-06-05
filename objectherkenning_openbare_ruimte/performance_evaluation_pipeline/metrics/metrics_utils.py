from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


class ObjectClass(Enum):
    person = 0
    license_plate = 1
    container = 2
    mobile_toilet = 3
    scaffolding = 4

    def __repr__(self):
        return self.value


class BoxSize:
    small: Tuple[float, float]
    medium: Tuple[float, float]
    large: Tuple[float, float]
    all: Tuple[float, float] = (0.0, 1.0)

    def __init__(self, bounds: Tuple[float, float] = (0.005, 0.01)):
        self.small = (0.0, bounds[0])
        self.medium = bounds
        self.large = (bounds[1], 1.0)

    @classmethod
    def from_objectclass(cls, object_class: ObjectClass):
        switch = {
            ObjectClass.person: (0.000665, 0.003397),
            ObjectClass.license_plate: (0.000108, 0.000436),
            ObjectClass.container: (0.003424, 0.022598),
            ObjectClass.mobile_toilet: (0.000854, 0.004376),
            ObjectClass.scaffolding: (0.010298, 0.125452),
        }
        return cls(switch.get(object_class))

    def to_dict(self, all_only: bool = False) -> Dict[str, Tuple[float, float]]:
        if all_only:
            return {"all": self.all}
        else:
            return {
                "small": self.small,
                "medium": self.medium,
                "large": self.large,
                "all": self.all,
            }

    def __repr__(self) -> str:
        return repr(self.medium)


def parse_labels(
    file_path: str,
) -> Tuple[List[int], List[Tuple[float, float, float, float]]]:
    """
     Parses a txt file with the following normalized format: [class x_center, y_center, width, height]

    Parameters
    ----------
    file_path The path to the labels file to be parsed.

    Returns (classes and bounding_boxes)
    -------

    """
    with open(file_path, "r") as f:
        lines = f.readlines()

    classes = [int(line.strip().split()[0]) for line in lines]
    bounding_boxes = [
        (
            float(line.strip().split()[1]),
            float(line.strip().split()[2]),
            float(line.strip().split()[3]),
            float(line.strip().split()[4]),
        )
        for line in lines
    ]
    return classes, bounding_boxes


def generate_binary_mask(
    bounding_boxes, image_width=3840, image_height=2160, consider_upper_half=False
):
    """
    Creates binary mask where all points inside the bounding boxes are 1, 0 otherwise.

    Parameters
    ----------
    bounding_boxes: list of bounding box coordinates
    image_width
    image_height
    consider_upper_half: only look at the upper half of the bounding boxes

    Returns
    -------

    """

    mask = np.zeros((image_height, image_width), dtype=bool)

    if len(bounding_boxes):
        bounding_boxes = np.array(bounding_boxes)
        y_min = (
            (bounding_boxes[:, 1] - bounding_boxes[:, 3] / 2) * image_height
        ).astype(int)
        x_min = (
            (bounding_boxes[:, 0] - bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        x_max = (
            (bounding_boxes[:, 0] + bounding_boxes[:, 2] / 2) * image_width
        ).astype(int)
        if consider_upper_half:
            y_max = (bounding_boxes[:, 1] * image_height).astype(int)
        else:
            y_max = (
                (bounding_boxes[:, 1] + bounding_boxes[:, 3] / 2) * image_height
            ).astype(int)
        for i in range(len(x_min)):
            mask[y_min[i] : y_max[i], x_min[i] : x_max[i]] = 1

    return mask
