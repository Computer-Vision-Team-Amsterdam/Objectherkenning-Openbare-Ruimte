import json
import os
from typing import Any, Dict, Iterable, List, Optional

import geopandas as gpd
import pandas as pd
import shapely.geometry as sg


def yolo_string_to_dict(yolo_annotation: str) -> Dict[str, Any]:
    """
    Convert a line of YOLO annotation text to a dict format.

    Expects `cat_id x_center y_center width height conf`.

    Returns `{"category": int, "confidence": float, "bbox": shapely.geometry.box}`.
    """
    values = yolo_annotation.split()

    if len(values) >= 6:
        cat_id, x_center, y_center, width, height, conf = map(float, values[:6])
    elif len(values) == 5:
        cat_id, x_center, y_center, width, height = map(float, values)
        conf = 1.0
    else:
        raise ValueError(f"Unrecognized annotation string format: {yolo_annotation}")

    bbox = sg.box(
        minx=x_center - width / 2,
        miny=y_center - height / 2,
        maxx=x_center + width / 2,
        maxy=y_center + height / 2,
    )

    data = {"category": int(cat_id), "confidence": conf, "bbox": bbox}

    return data


def yolo_file_to_dicts(annotation_file_path: str) -> List[Dict[str, Any]]:
    """
    Read a YOLO annotations file and return the annotations as a list of dicts.

    See `yolo_string_to_dict(..)` for details on the dict contents.
    """
    data = []

    with open(annotation_file_path, "r") as f:
        image_name = os.path.splitext(os.path.basename(annotation_file_path))[0]
        for line in f.readlines():
            line_data = yolo_string_to_dict(line)
            line_data["image_name"] = image_name
            data.append(line_data)

    return data


def oor_json_to_dicts(annotation_file_path: str) -> List[Dict[str, Any]]:
    """
    Read an OOR annotations file and return the annotations as a list of dicts.
    """
    data = []

    with open(annotation_file_path, "r") as f:
        json_content = json.load(f)
        image_file_name = json_content["image_file_name"]
        for detection in json_content["detections"]:
            yolo_bbox = detection["boundingBox"]
            bbox = sg.box(
                minx=yolo_bbox["x_center"] - yolo_bbox["width"] / 2,
                miny=yolo_bbox["y_center"] - yolo_bbox["height"] / 2,
                maxx=yolo_bbox["x_center"] + yolo_bbox["width"] / 2,
                maxy=yolo_bbox["y_center"] + yolo_bbox["height"] / 2,
            )

            line_data = {
                "category": detection["object_class"],
                "confidence": detection["confidence"],
                "bbox": bbox,
                "image_name": image_file_name,
            }
            data.append(line_data)

    return data


def read_annotations_folder(
    folder_path: str,
    file_type: str = ".txt",
    categories: Optional[Iterable[int]] = None,
    agnostic: bool = False,
) -> gpd.GeoDataFrame:
    """
    Convert all YOLO annotation files in a folder to GeoDataFrame with one annotation per row.

    The GeoDataFrame has columns `"image_name", "category", "confidence", "bbox"`.
    """
    if file_type not in (".txt", ".json"):
        print(f"Incorrect file_type, got {file_type}, expected .txt or .json.")
        return None

    data = []
    annotation_files = [
        file
        for file in os.listdir(folder_path)
        if os.path.splitext(file)[1] == file_type
    ]

    for file in annotation_files:
        if file_type == ".txt":
            data.extend(yolo_file_to_dicts(os.path.join(folder_path, file)))
        elif file_type == ".json":
            data.extend(oor_json_to_dicts(os.path.join(folder_path, file)))

    gdf = gpd.GeoDataFrame(
        data=data,
        columns=["image_name", "category", "confidence", "bbox"],
        geometry="bbox",
    )

    if categories is not None:
        gdf = gdf[gdf["category"].isin(categories)]

    if agnostic:
        gdf["category"] == 0

    return gdf


def read_coco_annotations(
    json_file: str, categories: Optional[Iterable[int]] = None
) -> gpd.GeoDataFrame:
    with open(json_file) as f:
        json_content = json.load(f)

    images_df = pd.DataFrame(json_content["images"]).set_index("id")
    images_df["image_file_name"] = [
        os.path.basename(file) for file in images_df["file_name"]
    ]
    images_df = images_df[["image_file_name"]]

    annotations_df = pd.DataFrame(json_content["annotations"]).set_index("id")
    annotations_df["category_id"] = annotations_df["category_id"] - 1
    if categories is not None:
        annotations_df = annotations_df[annotations_df["category_id"].isin(categories)]

    return annotations_df.join(images_df, on="image_id", how="left")


def generate_oor_json(detections: gpd.GeoDataFrame, output_folder: str):
    os.makedirs(output_folder, exist_ok=True)
    for image_file_name, idxs in detections.groupby("image_file_name").groups.items():
        json_file_name = os.path.join(
            output_folder, os.path.splitext(image_file_name)[0] + ".json"
        )
        coordinates: sg.Point = detections.loc[idxs[0], "geometry"]
        json_content = {
            "gps_data": {
                "latitude": coordinates.y,
                "longitude": coordinates.x,
            },
            "image_file_name": image_file_name,
            "detections": [],
        }
        for idx in idxs:
            minx, miny, maxx, maxy = detections.loc[idx, "bbox"].bounds
            json_content["detections"].append(
                {
                    "object_class": int(detections.loc[idx, "category_id"]),
                    "confidence": float(detections.loc[idx, "confidence"]),
                    "tracking_id": -1,
                    "boundingBox": {
                        "x_center": (minx + maxx) / 2,
                        "y_center": (miny + maxy) / 2,
                        "width": maxx - minx,
                        "height": maxy - miny,
                    },
                }
            )
        with open(json_file_name, "w") as f:
            json.dump(
                json_content,
                fp=f,
                indent=4,
            )
