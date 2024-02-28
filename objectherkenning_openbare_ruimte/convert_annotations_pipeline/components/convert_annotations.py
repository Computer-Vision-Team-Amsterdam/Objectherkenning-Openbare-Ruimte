import json
import logging
import os
import sys
from datetime import datetime
from typing import Any, Dict, List

import cv2
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from aml_interface.azure_logging import setup_azure_logging  # noqa: E402
from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)
settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
log_settings = ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)[
    "logging"
]
setup_azure_logging(log_settings, __name__)
aml_experiment_settings = settings["aml_experiment_details"]
logger = logging.getLogger("convert_annotations")


# Helper function to read YOLO annotations and convert them to COCO format
def yolo_to_coco(
    yolo_annotation: str, img_width: int, img_height: int
) -> Dict[str, Any]:
    class_id, x_center, y_center, width, height = map(float, yolo_annotation.split())
    class_id += 1  # Adjust class ID based on predefined mapping
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
    # Format bbox and area values with a consistent number of decimal places
    bbox = [
        round(x_min, 17),
        round(y_min, 17),
        round(norm_width, 17),
        round(norm_height, 17),
    ]
    area = round(norm_width * norm_height, 10)
    return {"category_id": int(class_id), "bbox": bbox, "area": area}


@command_component(
    name="convert_annotations",
    display_name="Converts annotations from YOLO to Azure COCO format.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def convert_annotations(
    input_old_folder: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_new_folder: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    datastore_name: str,
):
    """
    Pipeline step to convert annotations from YOLO to Azure COCO format.

    Parameters
    ----------
    input_old_folder:
        Path to the folder containing the annotations to convert.
    output_new_folder:
        Path to the folder containing the converted annotations.
    datastore_name: str
        Name of the datastore of the dataset.
    """

    # image_paths = find_image_paths(input_old_folder)
    logger.info(f"Input folder: {input_old_folder}")
    logger.info(f"Output folder: {output_new_folder}")
    logger.info(f"Datastore name: {datastore_name}")

    # Predefined categories mapping from YOLO to Azure COCO
    categories: List[Dict[str, Any]] = [
        {"id": 1, "name": "person"},
        {"id": 2, "name": "license plate"},
        {"id": 3, "name": "container"},
        {"id": 4, "name": "mobile toilet"},
        {"id": 5, "name": "scaffolding"},
        {"id": 6, "name": "noObjects"},
    ]

    # Initialize the JSON structure
    coco_json: Dict[str, Any] = {
        "images": [],
        "annotations": [],
        "categories": categories,
    }
    annotation_id: int = 1  # Unique ID for each annotation

    # Iterate over each image folder
    image_paths = find_image_paths(input_old_folder)
    for img_path in image_paths:
        print(f"img_path: {img_path}")
        annotation_file = os.path.splitext(img_path)[0] + ".txt"
        print(f"annotation_file: {annotation_file}")

        image_id: int = len(coco_json["images"]) + 1
        img = cv2.imread(img_path)
        height, width, _ = img.shape

        # An example of an img_path is:
        # "/mnt/azureml/cr/j/da2bd3655630436e9d1ad24e32103edb/cap/data-capability/
        # wd/images/testINPUT_input_old_folder/TMX7316010203-000335_pano_0000_008089/right.png"
        # We want folder_name = "images/test/INPUT_input_old_folder/TMX7316010203-000335_pano_0000_008089"
        # and file_name = "right.png"
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

        print(f"folder_name: {folder_name}")
        print(f"file_name: {file_name}")

        file_name_formatted = f"{folder_name}/{file_name}"
        coco_url = f"AmlDatastore://{datastore_name}/{file_name_formatted}"
        absolute_url = f"https://cvoamlweupgwapeg4pyiw5e7.blob.core.windows.net/{datastore_name}/{file_name_formatted.replace(' ', '%20')}"

        coco_json["images"].append(
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

        # Check if the annotation file exists
        if os.path.exists(annotation_file):
            with open(annotation_file, "r") as f:
                for line in f:
                    coco_annotation = yolo_to_coco(line, width, height)
                    # Construct annotation dict with the specified key order and formatted values
                    formatted_annotation = {
                        "id": annotation_id,
                        "category_id": coco_annotation["category_id"],
                        "image_id": image_id,
                        "area": coco_annotation["area"],
                        "bbox": coco_annotation["bbox"],
                    }
                    coco_json["annotations"].append(formatted_annotation)
                    annotation_id += 1

    """for folder_name in os.listdir(input_old_folder):
        print(f'folder_name: {folder_name}')
        folder_path = os.path.join(input_old_folder, folder_name)
        print(f'folder_path: {folder_path}')
        if os.path.isdir(folder_path):
            for file_name in os.listdir(folder_path):
                print(f'file_name: {file_name}')
                if file_name.endswith(".png"):
                    img_path = os.path.join(folder_path, file_name)
                    print(f'img_path: {img_path}')
                    annotation_file = os.path.splitext(img_path)[0] + ".txt"
                    print(f'annotation_file: {annotation_file}')

                    image_id: int = len(coco_json["images"]) + 1
                    img = cv2.imread(img_path)
                    height, width, _ = img.shape

                    file_name_formatted = (
                        f"{folder_name}/{file_name}"
                    )
                    coco_url = (
                        f"AmlDatastore://{datastore_name}/{file_name_formatted}"
                    )
                    absolute_url = f"https://cvoamlweupgwapeg4pyiw5e7.blob.core.windows.net/{datastore_name}/{file_name_formatted.replace(' ', '%20')}"

                    coco_json["images"].append(
                        {
                            "id": image_id,
                            "width": width,
                            "height": height,
                            "file_name": file_name_formatted,
                            "coco_url": coco_url,
                            "absolute_url": absolute_url,
                            "date_captured": datetime.now().strftime(
                                "%Y-%m-%dT%H:%M:%S.%fZ"
                            ),
                        }
                    )

                    # Check if the annotation file exists
                    if os.path.exists(annotation_file):
                        with open(annotation_file, "r") as f:
                            for line in f:
                                coco_annotation = yolo_to_coco(line, width, height)
                                # Construct annotation dict with the specified key order and formatted values
                                formatted_annotation = {
                                    "id": annotation_id,
                                    "category_id": coco_annotation["category_id"],
                                    "image_id": image_id,
                                    "area": coco_annotation["area"],
                                    "bbox": coco_annotation["bbox"],
                                }
                                coco_json["annotations"].append(formatted_annotation)
                                annotation_id += 1"""

    # Save to JSON file
    output_file = output_new_folder + "/annotations.json"
    with open(output_file, "w") as f:
        json.dump(coco_json, f, indent=4)

    print(f"JSON file created in: {output_file}")
