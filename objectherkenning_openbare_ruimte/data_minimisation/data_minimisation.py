import argparse
import os
import pathlib
import sys
from enum import Enum
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.data_minimisation import (  # noqa: E402
    blurring_tools,
)
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()


class Scenarios(Enum):
    A = True, False, False, True
    B = True, True, False, False
    C = True, False, True, False, True

    def __init__(
        self,
        blur_inside: bool,
        blur_outside: bool,
        crop: bool,
        draw_box: bool,
        fill_bg: bool = False,
    ):
        self.blur_inside = blur_inside
        self.blur_outside = blur_outside
        self.crop = crop
        self.draw_box = draw_box
        self.fill_bg = fill_bg


class DataMinimisation:
    """
    This class provides tools for data minimisation such as blurring and cropping following specified scenarios.
    """

    def __init__(self):
        self.settings = settings["data_minimisation"]

    @staticmethod
    def load_image_and_annotations(
        image_path, annotations_folder
    ) -> Tuple[npt.NDArray[np.int_], List[str]]:
        """
        Load an image and the corresponding annotations.

        Parameters
        ----------
        image path : str
            Path of the image.
        annotations_folder : str
            Path to folder where annotations can be found.

        Returns
        -------
        A tuple with the image as numpy array and annotations as list of strings.
        """
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)

        filename = pathlib.Path(image_path).stem
        label_file = os.path.join(annotations_folder, filename + ".txt")
        if os.path.isfile(label_file):
            with open(label_file, "r") as file:
                annotations = file.readlines()
        else:
            annotations = []

        return image, annotations

    def process_image(
        self,
        image: npt.NDArray[np.int_],
        yolo_annotations: List[str],
        scenario: Scenarios,
    ) -> npt.NDArray[np.int_]:
        """
        Process a single image following the specified data minimisation scenario.

        Supports a single target class (the object of interest) and a number of sensitive classes that should be blurred.

        Parameters
        ----------
        image : numpy.NDArray
            The image to process.
        yolo_annotations : list[str]
            A list of annotation strings in YOLO format.
        scenario : Scenarios instance
            The data minimisation scenario to apply.

        Returns
        -------
        The processed image.
        """
        image = image.copy()
        img_height, img_width, _ = image.shape

        for line in yolo_annotations:
            yolo_annotation = line.strip()
            class_id = int(yolo_annotation.split(sep=" ", maxsplit=1)[0])
            annotation_bounds = blurring_tools.yolo_annotation_to_bounds(
                yolo_annotation, img_height, img_width
            )
            if scenario.blur_inside and (
                class_id in self.settings["sensitive_classes"]
            ):
                image = blurring_tools.blur_inside_bounds(
                    image, annotation_bounds, self.settings["blur_kernel_size_inside"]
                )
            if scenario.blur_outside and (class_id == self.settings["target_class"]):
                image = blurring_tools.blur_outside_bounds(
                    image,
                    annotation_bounds,
                    self.settings["blur_kernel_size_outside"],
                    self.settings["blur_outside_padding"],
                )
            if scenario.crop and (class_id == self.settings["target_class"]):
                image = blurring_tools.crop_outside_bounds(
                    image,
                    annotation_bounds,
                    self.settings["crop_padding"],
                    scenario.fill_bg,
                )
            if scenario.draw_box and (class_id == self.settings["target_class"]):
                image = blurring_tools.draw_box_from_bounds(image, annotation_bounds)
        return image

    def process_folder(
        self,
        images_folder: str,
        annotations_folder: str,
        output_folder: str,
        image_format: str = "jpg",
    ):
        """
        Process all images in a given folder following all data minimisation scenarios, and saves the resulting images.

        Supports a single target class (the object of interest) and a number of sensitive classes that should be blurred. Scenarios are specified in the Scenarios class.

        Parameters
        ----------
        images_folder : str
            Folder with images to process.
        annotations_folder : str
            Path to folder where annotations can be found.
        output_folder : str
            Folder where processed images will be saved.
        image_format : str (default: 'jpg')
            Image file format.
        """
        pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

        images = pathlib.Path(images_folder).glob(f"*.{image_format}")
        for img_file in images:
            print(f"=== {img_file.stem} ===")
            image_raw, yolo_annotations = self.load_image_and_annotations(
                img_file.as_posix(), annotations_folder
            )

            if len(yolo_annotations) == 0:
                print("Nothing to blur!")
            else:
                yolo_annotations = sorted(yolo_annotations)
                for scenario in Scenarios:
                    print(f"--- Scenario {scenario.name} ---")
                    image = self.process_image(image_raw, yolo_annotations, scenario)
                    out_path = os.path.join(
                        output_folder, f"{img_file.stem}_scen_{scenario.name}.jpg"
                    )
                    cv2.imwrite(out_path, image)


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--images_folder", dest="images_folder", type=str)
    parser.add_argument("--labels_folder", dest="labels_folder", type=str)
    parser.add_argument("--output_folder", dest="output_folder", type=str)
    parser.add_argument("--image_format", dest="image_format", type=str)

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    data_minimisation = DataMinimisation()
    data_minimisation.process_folder(
        args.images_folder,
        args.labels_folder,
        args.output_folder,
        args.image_format,
    )
