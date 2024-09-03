import json
import os
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd
from pycocotools.coco import COCO

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.custom_coco_evaluator import (  # noqa: E402
    CustomCOCOeval,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    BoxSize,
    ObjectClass,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_image_stats import (  # noqa: E402
    EvaluateImageWise,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_pixel_stats import (  # noqa: E402
    EvaluatePixelWise,
)

DEFAULT_OBJECT_CLASSES = ObjectClass
DEFAULT_SENSITIVE_CLASSES = [
    ObjectClass.person,
    ObjectClass.license_plate,
]


class OOREvaluation:

    def __init__(
        self,
        ground_truth_base_folder: str,
        predictions_base_folder: str,
        predictions_image_shape: Tuple[int, int] = (3840, 2160),
        model_name: Union[str, None] = None,
        gt_annotations_rel_path: str = "labels",
        pred_annotations_rel_path: str = "labels",
        splits: Union[List[str], None] = ["train", "val", "test"],
        object_classes: Iterable[ObjectClass] = DEFAULT_OBJECT_CLASSES,
        sensitivate_classes: Iterable[ObjectClass] = DEFAULT_SENSITIVE_CLASSES,
        single_size_only: bool = False,
    ):
        self.ground_truth_base_folder = ground_truth_base_folder
        self.predictions_base_folder = predictions_base_folder
        self.predictions_image_shape = predictions_image_shape
        self.model_name = (
            model_name
            if model_name
            else os.path.basename(os.path.dirname(predictions_base_folder))
        )
        self.gt_annotations_rel_path = gt_annotations_rel_path
        self.pred_annotations_rel_path = pred_annotations_rel_path
        self.splits = splits
        self.object_classes = object_classes
        self.sensitivate_classes = sensitivate_classes
        self.single_size_only = single_size_only

    def tba_evaluation(
        self,
        save_results: bool = False,
        results_file: str = "tba_results.md",
        upper_half: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        split_names = self.splits if self.splits else [""]
        tba_results = dict()
        for split in split_names:
            print(
                f"Running TBA evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder = os.path.join(
                self.ground_truth_base_folder, self.gt_annotations_rel_path, split
            )
            prediction_folder = os.path.join(
                self.predictions_base_folder, self.pred_annotations_rel_path, split
            )
            evaluator = EvaluatePixelWise(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                image_shape=self.predictions_image_shape,
                upper_half=upper_half,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            tba_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.sensitivate_classes,
                single_size_only=self.single_size_only,
            )
        if save_results:
            evaluator.store_tba_results(
                results=tba_results,
                markdown_output_path=os.path.join(
                    self.predictions_base_folder, results_file
                ),
            )
        return tba_results

    def per_image_evaluation(self) -> Dict[str, Dict[str, float]]:
        split_names = self.splits if self.splits else [""]
        per_image_results = dict()
        for split in split_names:
            print(
                f"Running per-image evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder = os.path.join(
                self.ground_truth_base_folder, self.gt_annotations_rel_path, split
            )
            prediction_folder = os.path.join(
                self.predictions_base_folder, self.pred_annotations_rel_path, split
            )
            evaluator = EvaluateImageWise(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                image_shape=self.predictions_image_shape,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            per_image_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.object_classes, single_size_only=self.single_size_only
            )
        return per_image_results

    def coco_evaluation(
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
    ):
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

        return evaluation


def tba_result_to_df(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    def _cat_to_header(cat: str) -> str:
        parts = [p.capitalize().replace("All", "ALL") for p in cat.split(sep="_")]
        return " ".join(parts)

    models = list(results.keys())
    categories = list(results[models[0]].keys())
    header = ["Model", "Split"]
    header.extend([_cat_to_header(cat) for cat in categories])

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        data = [model_name, split]
        data.extend([results[model][cat]["recall"] for cat in categories])
        df.loc[len(df)] = data

    return df


def per_image_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:

    models = list(results.keys())
    categories = list(results[models[0]].keys())
    header = [
        "Model",
        "Split",
        "Object Class",
        "Size",
        "Precision",
        "Recall",
        "FPR",
        "FNR",
        "TNR",
    ]

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            (cat_name, size) = cat.rsplit(sep="_", maxsplit=1)
            data = [model_name, split, cat_name, size]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df
