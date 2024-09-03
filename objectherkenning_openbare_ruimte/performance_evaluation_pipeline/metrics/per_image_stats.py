from typing import Dict, Iterable, Set, Tuple

from cvtoolkit.datasets.yolo_labels_dataset import YoloLabelsDataset

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    BoxSize,
    ObjectClass,
)


class EvaluateImageWise:

    def __init__(
        self,
        ground_truth_path: str,
        predictions_path: str,
        image_shape: Tuple[int, int] = (3840, 2160),
    ):
        img_area = image_shape[0] * image_shape[1]
        if ground_truth_path.endswith(".json"):
            self.gt_dataset = YoloLabelsDataset.from_yolo_validation_json(
                yolo_val_json=ground_truth_path, image_shape=image_shape
            )
        else:
            self.gt_dataset = YoloLabelsDataset(
                folder_path=ground_truth_path, image_area=img_area
            )
        if predictions_path.endswith(".json"):
            self.pred_dataset = YoloLabelsDataset.from_yolo_validation_json(
                yolo_val_json=predictions_path, image_shape=image_shape
            )
        else:
            self.pred_dataset = YoloLabelsDataset(
                folder_path=predictions_path, image_area=img_area
            )
        self.gt_all = self._get_filename_set(self.gt_dataset)

    def collect_results_per_class_and_size(
        self,
        classes: Iterable[ObjectClass] = ObjectClass,
        single_size_only: bool = False,
    ) -> Dict[str, Dict[str, float]]:

        results = {}

        for target_class in classes:
            self.pred_dataset.reset_filter()
            predictions = self._get_filename_set(
                self.pred_dataset.filter_by_class(target_class.value)
            )

            box_sizes = BoxSize.from_objectclass(target_class).to_dict(single_size_only)

            for box_size_name, box_size in box_sizes.items():
                self.gt_dataset.reset_filter()
                ground_truth = self._get_filename_set(
                    self.gt_dataset.filter_by_class(
                        class_to_keep=target_class.value
                    ).filter_by_size_percentage(perc_to_keep=box_size)
                )

                results[f"{target_class.name}_{box_size_name}"] = self._compute_stats(
                    ground_truth, predictions, box_size_name
                )

        return results

    def _get_filename_set(self, yolo_dataset: YoloLabelsDataset) -> Set:
        labels = yolo_dataset.get_filtered_labels()
        return set(k for k, v in labels.items() if len(v) > 0)

    def _compute_stats(
        self, ground_truth: Set, predictions: Set, box_size_name: str
    ) -> Dict:
        is_all = box_size_name == "all"
        P = len(ground_truth)
        N = len(self.gt_all - ground_truth)
        PP = len(predictions)

        tp = len(ground_truth & predictions)
        fp = len(predictions - ground_truth)
        tn = len((self.gt_all - ground_truth) & (self.gt_all - predictions))
        fn = len(ground_truth - predictions)

        precision = tp / PP if PP > 0 else None
        recall = tp / P if P > 0 else None
        fpr = fp / N
        fnr = fn / P if P > 0 else None
        tnr = tn / N

        return {
            "precision": precision if is_all else None,
            "recall": recall,
            "fpr": fpr if is_all else None,
            "fnr": fnr,
            "tnr": tnr if is_all else None,
        }
