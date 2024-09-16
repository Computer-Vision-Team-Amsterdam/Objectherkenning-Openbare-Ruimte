# Adapted from: https://www.kaggle.com/code/siddharthkumarsah/convert-yolo-annotations-to-coco-pascal-voc?scriptVersionId=123233495&cellId=1

import json
import os
import pathlib
from typing import Dict, List, Tuple, Union

from PIL import Image

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (
    ObjectClass,
)

# Define the categories for the COCO dataset
CATEGORIES = [{"id": obj_cls.value, "name": obj_cls.name} for obj_cls in ObjectClass]


def convert_yolo_predictions_to_coco_json(
    predictions_dir: str,
    image_shape: Tuple[int, int],
    labels_rel_path: str = "labels",
    splits: Union[List[str], None] = ["train", "val", "test"],
    output_dir: Union[str, None] = None,
):
    if not splits:
        splits = [""]
    if not output_dir:
        output_dir = predictions_dir

    for split in splits:
        label_dir = os.path.join(predictions_dir, labels_rel_path, split)

        prediction_data = _convert_predictions_split(label_dir, image_shape)

        # Save the predictions to a JSON file
        with open(os.path.join(output_dir, f"coco_predictions_{split}.json"), "w") as f:
            f.write(json.dumps(prediction_data))


def _convert_predictions_split(
    label_dir: str, image_shape: Tuple[int, int]
) -> List[Dict]:
    prediction_data = []
    for pred_file in pathlib.Path(label_dir).glob("*.txt"):
        with open(pred_file) as f:
            for annotation in f.readlines():
                cat, xn, yn, wn, hn, score = map(float, annotation.strip().split()[0:6])
                width = wn * image_shape[0]
                height = hn * image_shape[1]
                x = (xn * image_shape[0]) - (width / 2)
                y = (yn * image_shape[1]) - (height / 2)
                prediction_data.append(
                    {
                        "image_id": pred_file.stem,
                        "category_id": int(cat),
                        "bbox": [x, y, width, height],
                        "score": score,
                    }
                )
    return prediction_data


def convert_yolo_dataset_to_coco_json(
    dataset_dir: str,
    splits: Union[List[str], None] = ["train", "val", "test"],
    output_dir: Union[str, None] = None,
):
    if not splits:
        splits = [""]
    if not output_dir:
        output_dir = dataset_dir

    for split in splits:
        image_dir = os.path.join(dataset_dir, "images", split)
        label_dir = os.path.join(dataset_dir, "labels", split)

        coco_dataset = _convert_dataset_split(image_dir, label_dir)

        # Save the COCO dataset to a JSON file
        with open(os.path.join(output_dir, f"coco_gt_{split}.json"), "w") as f:
            json.dump(coco_dataset, f)


def _convert_dataset_split(image_dir: str, label_dir: str) -> Dict:
    image_list: List[Dict] = []
    annotation_list: List[Dict] = []

    # Loop through the images in the input directory
    for image_file in os.listdir(image_dir):

        # Load the image and get its dimensions
        image_path = os.path.join(image_dir, image_file)
        image = Image.open(image_path)
        width, height = image.size

        # Add the image to the COCO dataset
        image_dict = {
            "id": image_file.split(".")[0],
            "width": width,
            "height": height,
            "file_name": image_file,
        }
        image_list.append(image_dict)

        # Load the bounding box annotations for the image
        annotation_file = os.path.join(label_dir, f'{image_file.split(".")[0]}.txt')
        if not os.path.isfile(annotation_file):
            continue

        with open(annotation_file) as f:
            annotations = f.readlines()

            # Loop through the annotations and add them to the COCO dataset
            for ann in annotations:
                cl, x, y, w, h = map(float, ann.strip().split()[0:5])
                x_min, y_min = int((x - w / 2) * width), int((y - h / 2) * height)
                x_max, y_max = int((x + w / 2) * width), int((y + h / 2) * height)
                ann_dict = {
                    "id": len(annotation_list),
                    "image_id": image_file.split(".")[0],
                    "category_id": int(cl),
                    "bbox": [x_min, y_min, x_max - x_min, y_max - y_min],
                    "area": (x_max - x_min) * (y_max - y_min),
                    "iscrowd": 0,
                }
                annotation_list.append(ann_dict)

    # Define the COCO dataset dictionary
    coco_dataset = {
        "info": {},
        "licenses": [],
        "categories": CATEGORIES,
        "images": image_list,
        "annotations": annotation_list,
    }

    return coco_dataset
