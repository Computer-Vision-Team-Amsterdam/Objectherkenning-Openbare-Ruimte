import argparse
import os
import pathlib
import sys
from enum import Enum
from typing import List, Tuple, Union

import cv2
import numpy as np
import numpy.typing as npt

sys.path.append(os.getcwd())

from objectherkenning_openbare_ruimte.inference_pipeline.source.output_image import (  # noqa: E402
    OutputImage,
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

    def _load_image_and_annotations(
        self,
        image_path: Union[str, os.PathLike],
        annotations_folder: Union[str, os.PathLike],
    ) -> None:
        """
        Load an image and the corresponding annotations.

        Parameters
        ----------
        image path : Union[str, os.PathLike]
            Path of the image.
        annotations_folder : Union[str, os.PathLike]
            Path to folder where annotations can be found.
        """
        self.image = OutputImage(cv2.imread(image_path, cv2.IMREAD_COLOR))

        filename = pathlib.Path(image_path).stem
        label_file = os.path.join(annotations_folder, filename + ".txt")

        classes = []
        bounds = []

        if os.path.isfile(label_file):
            with open(label_file, "r") as file:
                annotation_strings = file.readlines()
                classes = [
                    int(line.strip().split(sep=" ", maxsplit=1)[0])
                    for line in annotation_strings
                ]
                bounds = [
                    self.yolo_annotation_to_bounds(
                        line, (self.image.shape[1], self.image.shape[0])
                    )
                    for line in annotation_strings
                ]

        self.annotations = {
            "classes": np.array(classes),
            "bounds": np.array(bounds),
        }

    def _apply_scenario(
        self, scenario: Scenarios
    ) -> Union[npt.NDArray[np.int_], List[npt.NDArray[np.int_]]]:
        """Apply scenario to the image"""
        sensitive_idxs = np.where(
            np.in1d(self.annotations["classes"], self.settings["sensitive_classes"])
        )[0]
        target_idxs = np.where(
            np.in1d(self.annotations["classes"], self.settings["target_classes"])
        )[0]

        cropped_images = []

        if len(sensitive_idxs) > 0:
            if (len(sensitive_idxs) > 0) and scenario.blur_inside:
                self.image.blur_inside_boxes(
                    boxes=self.annotations["bounds"][sensitive_idxs],
                    blur_kernel_size=self.settings["blur_kernel_size_inside"],
                )
        if len(target_idxs) > 0:
            if scenario.blur_outside:
                self.image.blur_outside_boxes(
                    boxes=self.annotations["bounds"][target_idxs],
                    blur_kernel_size=self.settings["blur_kernel_size_outside"],
                    box_padding=self.settings["blur_outside_padding"],
                )
            if scenario.draw_box:
                self.image.draw_bounding_boxes(
                    boxes=self.annotations["bounds"][target_idxs],
                    categories=self.annotations["classes"][target_idxs],
                )
            if scenario.crop:
                cropped_images = self.image.crop_outside_boxes(
                    boxes=self.annotations["bounds"][target_idxs],
                    box_padding=self.settings["crop_padding"],
                    fill_bg=scenario.fill_bg,
                )

        if scenario.crop and not scenario.fill_bg:
            return cropped_images
        else:
            return self.image.get_image()

    def process_image(
        self,
        image_path: Union[str, os.PathLike],
        annotations_folder: Union[str, os.PathLike],
        scenario: Scenarios,
    ) -> Union[npt.NDArray[np.int_], List[npt.NDArray[np.int_]]]:
        """
        Process a single image following the specified data minimisation scenario.

        Supports a single target class (the object of interest) and a number of sensitive classes that should be blurred.

        Parameters
        ----------
        image_path : Union[str, os.PathLike]
            Path to the image file.
        annotations_folder : Union[str, os.PathLike]
            Folder where YOLO annotations for corresponding image can be found.
        scenario : Scenarios instance
            The data minimisation scenario to apply.

        Returns
        -------
        The processed image, or a list of images when cropping is applied without background filling.
        """
        self._load_image_and_annotations(
            image_path=image_path, annotations_folder=annotations_folder
        )

        if len(self.annotations["classes"]) == 0:
            print("No annotations found!")
            return self.image.get_image()
        else:
            return self._apply_scenario(scenario=scenario)

    def process_folder(
        self,
        images_folder: Union[str, os.PathLike],
        annotations_folder: Union[str, os.PathLike],
        output_folder: Union[str, os.PathLike],
        image_format: str = "jpg",
    ):
        """
        Process all images in a given folder following all data minimisation scenarios, and saves the resulting images.

        Supports a single target class (the object of interest) and a number of sensitive classes that should be blurred. Scenarios are specified in the Scenarios class.

        Parameters
        ----------
        images_folder : Union[str, os.PathLike]
            Folder with images to process.
        annotations_folder : Union[str, os.PathLike]
            Path to folder where annotations can be found.
        output_folder : Union[str, os.PathLike]
            Folder where processed images will be saved.
        image_format : str (default: 'jpg')
            Image file format.
        """
        pathlib.Path(output_folder).mkdir(exist_ok=True, parents=True)

        images = pathlib.Path(images_folder).glob(f"*.{image_format}")
        for img_file in images:
            print(f"=== {img_file.stem} ===")
            self._load_image_and_annotations(
                image_path=img_file, annotations_folder=annotations_folder
            )
            if len(self.annotations["classes"]) == 0:
                print("No annotations found!")
            else:
                for scenario in Scenarios:
                    print(f"--- Scenario {scenario.name} ---")
                    output_image = self._apply_scenario(scenario)
                    if isinstance(output_image, list):
                        for idx, crop in enumerate(output_image):
                            out_path = os.path.join(
                                output_folder,
                                f"{img_file.stem}_scen_{scenario.name}_crop_{idx + 1}.jpg",
                            )
                            cv2.imwrite(out_path, crop)
                    else:
                        out_path = os.path.join(
                            output_folder, f"{img_file.stem}_scen_{scenario.name}.jpg"
                        )
                        cv2.imwrite(out_path, output_image)

    @staticmethod
    def yolo_annotation_to_bounds(
        yolo_annotation: str, img_shape: Tuple[int, int]
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
