import os
from typing import List, Tuple

import cv2
import numpy as np
import numpy.typing as npt
import torch


def get_annotion_string_from_boxes(boxes):
    boxes = boxes.cpu()

    annotation_lines = []

    for box in boxes:
        cls = int(box.cls.squeeze())
        conf = float(box.conf.squeeze())
        tracking_id = int(box.id.squeeze()) if box.is_track else -1
        yolo_box_str = " ".join([f"{x:.6f}" for x in box.xywhn.squeeze()])
        annotation_lines.append(f"{cls} {yolo_box_str} {conf:.6f} {tracking_id}")
    return "\n".join(annotation_lines)


def draw_bounding_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[int, int, int, int]],
    colours: List[Tuple[int, int, int]] = [(0, 0, 255)],
    box_padding: int = 0,
    line_thickness: int = 2,
) -> npt.NDArray[np.int_]:
    """
    Draw the given bounding box(es).

    Parameters
    ----------
    image : numpy.ndarray
        The image to draw on.
    boxes : List[Tuple[int, int, int, int]]
        Bounding box(es) to draw, in the format (xmin, ymin, xmax, ymax).
    colours : List[Tuple[int, int, int]] (default: [(0, 0, 255)])
        Optional: list of colours for each bounding box, in the format (255, 255, 255)
    box_padding : int (default: 0)
        Optional: increase box by this amount of pixels before drawing.
    line_thickness : int (default: 2)
        Line thickness for the bounding box.

    Returns
    -------
    numpy.ndarray
        The image with drawn bounding box.
    """
    img_height, img_width, _ = image.shape

    if len(colours) < len(boxes):
        difference = len(boxes) - len(colours)
        colours.extend([colours[-1]] * difference)

    for colour, box in zip(colours, boxes):
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Drawing: {(x_min, y_min)} -> {(x_max, y_max)} in colour {colour}")
        image = cv2.rectangle(
            image, (x_min, y_min), (x_max, y_max), colour, thickness=line_thickness
        )
    return image


def blur_inside_boxes(
    image: npt.NDArray[np.int_],
    boxes: List[Tuple[int, int, int, int]],
    blur_kernel_size: int = 165,
    box_padding: int = 0,
) -> npt.NDArray[np.int_]:
    """
    Apply GaussianBlur with given kernel size to the area given by the bounding box(es).

    Parameters
    ----------
    image : numpy.ndarray
        The image to blur.
    boxes : List[Tuple[int, int, int, int]]
        Bounding box(es) of the area(s) to blur, in the format (xmin, ymin, xmax, ymax).
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

    for box in boxes:
        x_min, y_min, x_max, y_max = map(int, box)

        x_min = max(0, x_min - box_padding)
        y_min = max(0, y_min - box_padding)
        x_max = min(img_width, x_max + box_padding)
        y_max = min(img_height, y_max + box_padding)

        # print(f"Blurring inside: {(x_min, y_min)} -> {(x_max, y_max)}")
        area_to_blur = image[y_min:y_max, x_min:x_max]
        blurred = cv2.GaussianBlur(
            area_to_blur, (blur_kernel_size, blur_kernel_size), 0
        )
        image[y_min:y_max, x_min:x_max] = blurred

    return image


def process_results(results, sensitive_classes, target_classes, images_dir, labels_dir):
    # TODO: Differentiate between inference for pre-labeling and use-case inference
    for r in results:
        result = r.cpu()
        boxes = result.boxes.numpy()

        print(f"Boxes classes: {boxes.cls}")
        print(f"Boxes: {boxes}")
        target_idxs = np.where(np.in1d(boxes.cls, target_classes))[0]

        image = result.orig_img.copy()
        sensitive_idxs = np.where(np.in1d(boxes.cls, sensitive_classes))[0]

        # Blur sensitive data
        sensitive_bounding_boxes = boxes[sensitive_idxs].xyxy
        image = blur_inside_boxes(image, sensitive_bounding_boxes)

        # Draw annotation boxes
        target_bounding_boxes = boxes[target_idxs].xyxy
        image = draw_bounding_boxes(image, target_bounding_boxes)

        # Save image
        image_path = result.path  # Path of the input image
        image_name = os.path.basename(image_path)
        image_file = os.path.join(images_dir, image_name)
        print(f"Folder path: {images_dir}")
        print(f"Save path: {image_file}")
        cv2.imwrite(image_file, image)
        print("Saved image.")

        # Save annotation
        annotation_str = get_annotion_string_from_boxes(boxes[target_idxs])
        annotation_path = os.path.join(labels_dir, f"{image_name}.txt")
        with open(annotation_path, "w") as f:
            f.write(annotation_str)
        print("Saved annotation.")


def process_batches(
    model,
    inference_params,
    image_paths,
    batch_size,
    images_dir,
    labels_dir,
    sensitive_classes,
    target_classes,
):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        inference_params["source"] = batch_paths
        try:
            batch_results = model(**inference_params)
            process_results(
                batch_results, sensitive_classes, target_classes, images_dir, labels_dir
            )
            torch.cuda.empty_cache()  # Clear unused memory

        except RuntimeError as e:
            if "out of memory" in str(e):
                print("Out of memory with batch size of {}".format(batch_size))
                if batch_size > 1:
                    new_batch_size = batch_size // 2
                    print("Trying smaller batch size: {}".format(new_batch_size))
                    return process_batches(
                        model, inference_params, image_paths, new_batch_size
                    )
                else:
                    raise RuntimeError("Out of memory with the smallest batch size")
            else:
                raise e


def process_tracking_results(results, labels_dir, tracking_classes):
    for r in results:
        result = r.cpu()
        image_path = result.path  # Path of the input image
        image_name = os.path.basename(image_path)
        label_file = os.path.join(labels_dir, f"{os.path.splitext(image_name)[0]}.txt")

        with open(label_file, "w") as f:
            for box in result.boxes:
                class_id = int(box.cls)
                if class_id in tracking_classes:
                    bbox = box.xywhn.tolist()[0]
                    track_id = box.id if hasattr(box, "id") else None

                    x_center, y_center, width, height = (
                        float(bbox[0]),
                        float(bbox[1]),
                        float(bbox[2]),
                        float(bbox[3]),
                    )
                    bbox_str = f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                    conf_score = float(box.conf)
                    conf_score_str = f"{conf_score:.6f}"

                    if track_id is not None:
                        line = (
                            f"{class_id} {bbox_str} {conf_score_str} {int(track_id)}\n"
                        )
                    else:
                        line = f"{class_id} {bbox_str} {conf_score_str} -1\n"

                    f.write(line)

    print("Tracking complete. Results saved.")
