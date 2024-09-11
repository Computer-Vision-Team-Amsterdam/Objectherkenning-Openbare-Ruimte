import os
from typing import Dict, Iterable, List, Tuple, Union

import pandas as pd

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    ObjectClass,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_image_stats import (  # noqa: E402
    EvaluateImageWise,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_pixel_stats import (  # noqa: E402
    EvaluatePixelWise,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.yolo_to_coco import (  # noqa: E402
    convert_yolo_dataset_to_coco_json,
    convert_yolo_predictions_to_coco_json,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.run_custom_coco_eval import (  # noqa: E402
    run_custom_coco_eval,
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
        output_folder: Union[str, None] = None,
        ground_truth_image_shape: Tuple[int, int] = (3840, 2160),
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
        self.output_folder = output_folder
        self.ground_truth_image_shape = ground_truth_image_shape
        self.predictions_image_shape = predictions_image_shape
        self.model_name = (
            model_name
            if model_name
            else os.path.basename(os.path.dirname(predictions_base_folder))
        )
        self.gt_annotations_rel_path = gt_annotations_rel_path
        self.pred_annotations_rel_path = pred_annotations_rel_path
        self.splits = splits if splits else [""]
        self.object_classes = object_classes
        self.sensitivate_classes = sensitivate_classes
        self.single_size_only = single_size_only

    def _get_folders_for_split(self, split: str) -> Tuple[str, str]:
        ground_truth_folder = os.path.join(
            self.ground_truth_base_folder, self.gt_annotations_rel_path, split
        )
        prediction_folder = os.path.join(
            self.predictions_base_folder, self.pred_annotations_rel_path, split
        )
        return ground_truth_folder, prediction_folder

    def tba_evaluation(
        self,
        upper_half: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        tba_results = dict()
        for split in self.splits:
            print(
                f"Running TBA evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
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
        return tba_results

    def per_image_evaluation(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        per_image_results = dict()
        for split in self.splits:
            print(
                f"Running per-image evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
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

    def coco_evaluation(self) -> Dict[str, Dict[str, Dict[str, float]]]:
        custom_coco_result: Dict[str, Dict[str, Dict[str, float]]] = dict()
        target_classes = {"all": [obj_cls.value for obj_cls in self.object_classes]}
        for obj_cls in self.object_classes:
            target_classes[obj_cls.name] = [obj_cls.value]

        if not self.output_folder:
            gt_output_dir = self.ground_truth_base_folder
        else:
            gt_output_dir = self.output_folder

        convert_yolo_dataset_to_coco_json(
            dataset_dir=self.ground_truth_base_folder,
            splits=self.splits,
            output_dir=gt_output_dir,
        )

        if not self.output_folder:
            pred_output_dir = self.predictions_base_folder
        else:
            pred_output_dir = self.output_folder

        convert_yolo_predictions_to_coco_json(
            predictions_dir=self.predictions_base_folder,
            image_shape=self.ground_truth_image_shape,
            labels_rel_path=self.pred_annotations_rel_path,
            splits=self.splits,
            output_dir=pred_output_dir,
        )

        for split in self.splits:
            gt_json = os.path.join(gt_output_dir, f"coco_gt_{split}.json")
            pred_json = os.path.join(pred_output_dir, f"coco_predictions_{split}.json")

            key = f"{self.model_name}_{split if split != '' else 'all'}"
            custom_coco_result[key] = dict()

            for target_cls_name, target_cls in target_classes.items():
                print(
                    f"Running custom COCO evaluation for {self.model_name} / {split if split != '' else 'all'} / {target_cls_name}"
                )
                eval = run_custom_coco_eval(
                    coco_annotations_json=gt_json,
                    coco_predictions_json=pred_json,
                    predicted_img_shape=self.ground_truth_image_shape,
                    class_ids=target_cls,
                    print_summary=False,
                )
                subkey = target_cls_name
                custom_coco_result[key][subkey] = eval
        return custom_coco_result


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
        data.extend([str(results[model][cat]["recall"]) for cat in categories])
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
            data.extend([str(val) for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def custom_coco_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:

    models = list(results.keys())
    categories = list(results[models[0]].keys())
    header = ["Model", "Split", "Object Class"]
    header.extend(list(results[models[0]][categories[0]].keys()))

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            data = [model_name, split, cat]
            data.extend([str(val) for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df
