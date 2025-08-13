import argparse
import copy
import json
import os
import shutil
from typing import List

import cv2
from yolo_model_development_kit.inference_pipeline.source.output_image import (
    OutputImage,
)

class_to_name = {
    2: "Container",
    3: "Dixie",
    4: "Scaffolding",
}


def validate_detections(
    detections_folder: str, images_folder: str, resume: bool = False
):
    print(
        "\n======\n"
        "Validate OOR detections. Inspect each detection visually, "
        "and press {[1], [SPACE], [ENTER]} to accept, or {[2], [F]} to reject. "
        "Rejected detections will be moved to the new subfolder 'fp'. "
        "Press [ESC] to quit."
        "\n======\n"
    )

    cv2.namedWindow("Validate detections OOR")

    progress_file = os.path.join(detections_folder, "validated_detections.txt")

    print(f"Scanning metadata in {detections_folder}")
    detection_metadata_files = get_detection_metadata_file_paths(detections_folder)
    print(f" - Found {len(detection_metadata_files)} detection metadata files.")

    if resume:
        done_files = []
        if os.path.exists(progress_file):
            with open(progress_file, "r") as f:
                done_files = f.read().splitlines()
        done_files = [f for f in done_files if f in detection_metadata_files]
        print(f" - Found {len(done_files)} detection files marked as done.")
        detection_metadata_files = [
            df for df in detection_metadata_files if df not in done_files
        ]
        print(f"{len(detection_metadata_files)} still to do.")
    else:
        with open(progress_file, "w") as f:
            pass

    n_det = len(detection_metadata_files)

    fp_det_dir = os.path.join(detections_folder, "fp")
    fp_img_dir = os.path.join(images_folder, "fp")
    os.makedirs(fp_det_dir, exist_ok=True)
    os.makedirs(fp_img_dir, exist_ok=True)

    i = 0

    for detection_file in detection_metadata_files:
        with open(detection_file, "r") as f:
            i += 1
            json_content = json.load(f)

            is_fp = False
            fp_json_content = copy.deepcopy(json_content)
            fp_json_content["detections"].clear()

            img_name = json_content["image_file_name"]
            img_file = os.path.join(images_folder, img_name)
            if not os.path.isfile(img_file):
                raise FileNotFoundError(f"Image file {img_file} not found.")

            raw_image = cv2.imread(img_file)
            img_width, img_height = raw_image.shape[1], raw_image.shape[0]

            for detection in json_content["detections"]:
                x_min = (
                    detection["boundingBox"]["x_center"]
                    - detection["boundingBox"]["width"] / 2
                )
                x_max = (
                    detection["boundingBox"]["x_center"]
                    + detection["boundingBox"]["width"] / 2
                )
                y_min = (
                    detection["boundingBox"]["y_center"]
                    - detection["boundingBox"]["height"] / 2
                )
                y_max = (
                    detection["boundingBox"]["y_center"]
                    + detection["boundingBox"]["height"] / 2
                )
                x_min = int(x_min * img_width)
                x_max = int(x_max * img_width)
                y_min = int(y_min * img_height)
                y_max = int(y_max * img_height)

                bbox = [x_min, y_min, x_max, y_max]
                obj_class = detection["object_class"]
                name = class_to_name[obj_class]

                image = OutputImage(raw_image.copy())
                image.draw_bounding_boxes(
                    boxes=[bbox], categories=[obj_class], tracking_ids=[name]
                )

                # cv2.moveWindow("Validate detections OOR", 275, 100)
                cv2.setWindowTitle(
                    "Validate detections OOR", f"Validating detection {i}/{n_det}"
                )
                cv2.imshow("Validate detections OOR", image.get_image())

                # The function waitKey waits for a key event infinitely (when delay<=0)
                k = None
                pos_values = [13, 32, 3, 83, 49, 102, 50, 27]
                while k not in pos_values:
                    k = cv2.waitKey(0)
                    if k in [13, 32, 3, 83, 49]:
                        # [enter] or [space] or [>] or [1]: true positive
                        continue
                    elif k in [102, 50]:  # [f] or [2]: false positive
                        is_fp = True
                        json_content["detections"].remove(detection)
                        fp_json_content["detections"].append(detection)
                        continue
                    elif k == 27:  # [esc] to exit the program
                        print("Exiting")
                        cv2.destroyAllWindows()
                        return 0
                    else:
                        print("Key not valid...")

            if is_fp:
                fp_det_file = os.path.join(fp_det_dir, os.path.basename(detection_file))
                if len(json_content["detections"]) > 0:
                    with open(detection_file, "w") as f:
                        json.dump(json_content, f, indent=4)
                    with open(fp_det_file, "w") as f:
                        json.dump(fp_json_content, f, indent=4)
                else:
                    shutil.move(detection_file, fp_det_dir)
                    shutil.move(img_file, fp_img_dir)

            with open(progress_file, "a") as f:
                f.write(detection_file + "\n")

    print("All detections validated.")


def get_detection_metadata_file_paths(
    root_folder: str,
    file_type: str = ".json",
    ignore_folders: List[str] = ["fp"],
) -> List[str]:
    """
    List all files with a given file_type (default: .json) in root_folder
    recursively. Optional ignore_folders will be skipped. Returns a sorted list.

    Parameters
    ----------
    root_folder : str
        Root folder
    file_type : str = ".json"
        Type of file to filter by
    ignore_folders: List[str] = ["fp"]
        List of folder names that will be skipped

    Returns
    -------
    List[str]
        Sorted list of file paths
    """
    files = []
    for dirpath, dirnames, filenames in os.walk(root_folder, topdown=True):
        dirnames[:] = [d for d in dirnames if d not in ignore_folders]
        for filename in filenames:
            if filename.endswith(file_type):
                filepath = os.path.join(dirpath, filename)
                files.append(filepath)
    return sorted(files)


def main():
    parser = argparse.ArgumentParser(
        description="Validate OOR detections from landing zone.",
        epilog="Download detection_metadata and images locally, set the the top level folder as input and specify the date of detections.",
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="e.g. /home/user/data/landing_zone",
    )
    parser.add_argument("--date", type=str, required=True, help="e.g. 2025-07-28")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Use flag to resume from previous validation session.",
    )
    args = parser.parse_args()

    input_folder = args.input_folder
    date = args.date
    resume = args.resume

    detections_folder = os.path.join(input_folder, "detection_metadata", date)
    images_folder = os.path.join(input_folder, "images", date)

    validate_detections(detections_folder, images_folder, resume)


if __name__ == "__main__":
    main()
