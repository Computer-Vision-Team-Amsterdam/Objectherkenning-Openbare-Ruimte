import json
import os
from datetime import datetime
from typing import Any, Dict, List

import cv2

# Input paths
input_folder = "../../output"
output_json = "../../output.json"
dataset_split_folder = "test"  # Change to 'train', 'val' or 'test' as needed

# Predefined categories mapping from YOLO to Azure COCO
categories: List[Dict[str, Any]] = [
    {"id": 1, "name": "person"},
    {"id": 2, "name": "license plate"},
    {"id": 3, "name": "container"},
    {"id": 4, "name": "mobile toilet"},
    {"id": 5, "name": "scaffolding"},
    {"id": 6, "name": "noObjects"},
]


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


# Initialize the JSON structure
coco_json: Dict[str, Any] = {"images": [], "annotations": [], "categories": categories}
annotation_id: int = 1  # Unique ID for each annotation

# Iterate over each image folder
for folder_name in os.listdir(input_folder):
    folder_path = os.path.join(input_folder, folder_name)
    if os.path.isdir(folder_path):
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".png"):
                img_path = os.path.join(folder_path, file_name)
                annotation_file = os.path.splitext(img_path)[0] + ".txt"

                image_id: int = len(coco_json["images"]) + 1
                img = cv2.imread(img_path)
                height, width, _ = img.shape

                file_name_formatted = (
                    f"images/{dataset_split_folder}/{folder_name}/{file_name}"
                )
                coco_url = (
                    f"AmlDatastore://converted_old_dataset_oor/{file_name_formatted}"
                )
                absolute_url = f"https://cvoamlweupgwapeg4pyiw5e7.blob.core.windows.net/converted-old-dataset-oor/{file_name_formatted.replace(' ', '%20')}"

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

# Save to JSON file
with open(output_json, "w") as f:
    json.dump(coco_json, f, indent=4)

print(f"JSON file created: {output_json}")
