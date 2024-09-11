import json
from typing import Dict, List, Tuple

from pycocotools.coco import COCO

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.custom_coco_evaluator import (  # noqa: E402
    CustomCOCOeval,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    BoxSize,
    ObjectClass,
)


def run_custom_coco_eval(
    coco_annotations_json: str,
    coco_predictions_json: str,
    predicted_img_shape: Tuple[int, int],
    class_ids: List[int] = [0, 1, 2, 3, 4],
    class_labels: List[str] = [
        "person",
        "license plate",
        "container",
        "mobile toilet",
        "scaffolding",
    ],
    print_summary: bool = True,
    precision: int = 3,
) -> Dict[str, float]:
    """
    Runs COCO evaluation on the output of YOLO validation

    Parameters
    ----------
    coco_annotations_json: annotations in the COCO format compatible with yolov5. Comes from the metadata pipeline
    coco_predictions_json: predictions in COCO format of the yolov5 run.
    metrics_metadata: info about image sizes and areas for sanity checks.

    Returns
    -------

    """
    COCO_gt = COCO(coco_annotations_json)  # init annotations api
    try:
        COCO_dt = COCO_gt.loadRes(coco_predictions_json)  # init predictions api
    except FileNotFoundError:
        raise Exception(
            f"The specified file '{coco_predictions_json}' was not found."
            f"The file is created at the above path if you run yolo validation with"
            f"the --save-json flag enabled."
        )
    evaluation = CustomCOCOeval(COCO_gt, COCO_dt, "bbox")

    # Opening JSON file
    with open(coco_annotations_json) as f:
        data = json.load(f)

    height = data["images"][0]["height"]
    width = data["images"][0]["width"]
    if width != predicted_img_shape[0] or height != predicted_img_shape[1]:
        print(
            f"You're trying to run evaluation on images of size {width} x {height}, "
            "but the coco annotations have been generated from images of size "
            f"{predicted_img_shape[0]} x {predicted_img_shape[1]}."
            "Why is it a problem? Because the coco annotations that the metadata produces and the "
            " *_predictions.json produced by the yolo run are both in absolute format,"
            "so we must compare use the same image sizes."
            "Solutions: 1. Use images for validation that are the same size as the ones you used for the "
            "annotations. 2. Re-compute the coco_annotations_json using the right image shape."
        )

    image_names = [image["id"] for image in data["images"]]
    evaluation.params.imgIds = image_names  # image IDs to evaluate
    evaluation.params.catIds = class_ids
    class_labels = [class_labels[i] for i in class_ids]
    evaluation.params.catLbls = class_labels

    img_area = height * width

    areaRng = []
    for areaRngLbl in evaluation.params.areaRngLbl:
        aRng = {"areaRngLbl": areaRngLbl}
        for obj_cls in ObjectClass:
            box = BoxSize.from_objectclass(obj_cls).__getattribute__(areaRngLbl)
            aRng[obj_cls.value] = (box[0] * img_area, box[1] * img_area)
        areaRng.append(aRng)

    evaluation.params.areaRng = areaRng

    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize(print_summary=print_summary)

    return _stats_to_dict(evaluation, precision)


def _stats_to_dict(eval: CustomCOCOeval, precision: int) -> Dict:
    keys = [
        "AP@50-95_all",
        "AP@75_all",
        "AP@50_all",
        "AP@50_small",
        "AP@50_medium",
        "AP@50_large",
        "AR@50-95_all",
        "AR@75_all",
        "AR@50_all",
        "AR@50_small",
        "AR@50_medium",
        "AR@50_large",
    ]
    values = eval.stats
    return {
        key: round(value, precision) if value else None
        for key, value in zip(keys, values)
    }
