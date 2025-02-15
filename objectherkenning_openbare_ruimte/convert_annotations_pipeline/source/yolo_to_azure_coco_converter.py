import json
import logging
import os
from datetime import datetime
from typing import Any, Dict

import cv2
from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

logger = logging.getLogger("convert_annotations_pipeline")


class YoloToAzureCocoConverter:
    """
    Converts a YOLO annotation dataset to Azure COCO format.
    """

    def __init__(
        self,
        input_folder: str,
        output_folder: str,
        datastore_name: str,
        image_storage_account: str,
        categories_file: str,
        separate_labels: bool = False,
        label_folder: str = None,
    ):
        """
        Parameters
        ----------
        input_folder: str
            Path to the folder containing the YOLO annotations.
        output_folder: str
            Path to the folder where the Azure COCO annotations will be stored.
        datastore_name: str
            Name of the datastore to be used in the COCO file URLs.
        image_storage_account: str
            Name of the storage account to be used in the COCO file URLs.
        categories_file: str
            Path to the JSON file containing the categories.
        """
        self.input_folder = input_folder
        self.output_folder = output_folder
        self.datastore_name = datastore_name
        self.image_storage_account = image_storage_account
        self.separate_labels = separate_labels
        self.label_folder = label_folder if separate_labels else input_folder
        self._load_categories(categories_file)
        self.coco_json = {
            "images": [],
            "annotations": [],
            "categories": self.categories,
        }
        self.annotation_id = 1

    def _load_categories(self, categories_file: str):
        """
        Loads categories from a JSON file.

        Parameters
        ----------
        categories_file: str
            Path to the JSON file containing the categories.
        """
        with open(categories_file, "r") as f:
            self.categories = json.load(f)

    @staticmethod
    def _parse_image_path(img_path: str) -> tuple:
        """
        Parses the image path to extract the folder and file names.

        Parameters
        ----------
        img_path: str
            The path of the image.

        Returns
        -------
        tuple
            The folder name and file name.

        Example
        -------
        img_path = "/mnt/azureml/cr/j/da2bd3655630436e9d1ad24e32103edb/cap/data-capability/
        wd/images/test/INPUT_input_old_folder/TMX7316010203-000335_pano_0000_008089/right.png"

        folder_name = "images/test/INPUT_input_old_folder/TMX7316010203-000335_pano_0000_008089"
        file_name = "right.png"
        """

        input_folder_start = (
            img_path.find("/INPUT_") + 1
        )  # +1 to exclude the leading '/'
        if input_folder_start > 0:
            # Find the next slash after /INPUT_{some_name}/ to determine the end of the base directory
            base_directory_end = img_path.find("/", input_folder_start)
            # Extract everything after the base directory as folder_name
            folder_name = img_path[
                base_directory_end + 1 : img_path.rfind("/")
            ]  # Exclude the file name itself
        else:
            # Fallback if /INPUT_{some_name}/ is not found, use the old method or handle the error
            folder_name = img_path.split("/")[-2]

        file_name = img_path.split("/")[-1]
        return folder_name, file_name

    @staticmethod
    def _yolo_to_coco(
        yolo_annotation: str, img_width: int, img_height: int
    ) -> Dict[str, Any]:
        """
        Converts a single YOLO annotation to COCO format.

        Parameters
        ----------
        yolo_annotation: str
            One line of YOLO annotation.
        img_width: int
            Width of the image.
        img_height: int
            Height of the image.

        Returns
        -------
        dict
            The annotation in COCO format.
        """
        # Split the annotation and select only the first 5 values
        values = yolo_annotation.split()[:5]
        class_id, x_center, y_center, width, height = map(float, values)

        class_id += 1  # Azure COCO expects classes to start from 1
        x_center, y_center, width, height = (
            x_center * img_width,
            y_center * img_height,
            width * img_width,
            height * img_height,
        )
        # Normalize bbox (COCO format expects top left x, top left y, width, height)
        x_min = (x_center - width / 2) / img_width
        y_min = (y_center - height / 2) / img_height
        norm_width = width / img_width
        norm_height = height / img_height
        # Azure COCO expects bbox to be rounded to 17 decimal places
        # and area to be rounded to 10 decimal places
        bbox = [
            round(x_min, 17),
            round(y_min, 17),
            round(norm_width, 17),
            round(norm_height, 17),
        ]
        area = round(norm_width * norm_height, 10)

        return {"category_id": int(class_id), "bbox": bbox, "area": area}

    def convert(self):
        """
        Converts the YOLO annotations in the input folder to Azure COCO format and saves them in the output folder.
        """
        img_paths = find_image_paths(self.input_folder)

        for img_path in img_paths:
            image_id = len(self.coco_json["images"]) + 1
            img = cv2.imread(img_path)
            height, width, _ = img.shape

            folder_name, file_name = YoloToAzureCocoConverter._parse_image_path(
                img_path
            )
            file_name_formatted = f"{folder_name}/{file_name}"
            coco_url = f"AmlDatastore://{self.datastore_name}/{file_name_formatted}"
            absolute_url = f"https://{self.image_storage_account}.blob.core.windows.net/{self.datastore_name}/{file_name_formatted.replace(' ', '%20')}"

            self.coco_json["images"].append(
                {
                    "id": image_id,
                    "width": width,
                    "height": height,
                    "file_name": file_name_formatted,
                    "coco_url": coco_url,
                    "absolute_url": absolute_url,
                    "date_captured": datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%fZ"),
                }
            )

            if self.separate_labels:
                annotation_file = os.path.join(
                    self.label_folder,
                    os.path.splitext(os.path.basename(img_path))[0] + ".txt",
                )
            else:
                annotation_file = os.path.splitext(img_path)[0] + ".txt"

            if os.path.exists(annotation_file):
                with open(annotation_file, "r") as f:
                    for line in f:
                        coco_annotation = YoloToAzureCocoConverter._yolo_to_coco(
                            line, width, height
                        )
                        # Construct annotation dict with the specified key order and formatted values
                        formatted_annotation = {
                            "id": self.annotation_id,
                            "category_id": coco_annotation["category_id"],
                            "image_id": image_id,
                            "area": coco_annotation["area"],
                            "bbox": coco_annotation["bbox"],
                        }
                        self.coco_json["annotations"].append(formatted_annotation)
                        self.annotation_id += 1

        output_file = os.path.join(self.output_folder, "annotations_new.json")
        logger.info(f"Categories: {self.categories}")
        with open(output_file, "w") as f:
            json.dump(self.coco_json, f, indent=4)
