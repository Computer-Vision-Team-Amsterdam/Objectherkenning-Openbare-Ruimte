from typing import Dict, Iterable, Optional, Set, Tuple

import numpy as np
from cvtoolkit.datasets.yolo_labels_dataset import YoloLabelsDataset

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    BoxSize,
    ObjectClass,
)


class PerImageEvaluator:
    """
    This class is used to run per-image evaluation over a dataset of ground
    truth and prediction labels. For each object class and bounding box size
    (small, medium, large) it will compute precision, recall, false positive
    rate, false negative rate, and true negative rate based on the per-image
    accuracy of the predictions.

    Per-image accuracy here means that we check for each image that contains at
    least one ground truth annotation for a specific class, whether it also has
    at least one prediction, regardless of whether the bounding boxes overlap or
    not.

    For the different bounding box sizes only recall can be computed. Precision
    is only meaningful aggregated over all bounding box sizes.

    Parameters
    ----------
        ground_truth_path: str
            Path to ground truth annotations, either as a folder with YOLO .txt
            annotation files, or as a COCO JSON file.
        predictions_path: str
            Path to ground truth annotations, either as a folder with YOLO .txt
            annotation files, or as a COCO JSON file.
        image_shape: Tuple[int, int] = (3840, 2160)
            Shape of the images. Since YOLO .txt annotations contain bounding
            box dimensions as fraction of the image shape, the pixel dimensions
            are less important as long as the ratio is preserved. Higher pixel
            resolution might lead to better precision at the cost of higher
            computation time.
            When annotations are provided as COCO JSON, it is important that the
            shape provided here is equal to the shape in the ground truth
            annotation JSON.
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, all predictions will be used.
        decimals: int = 3
            Round statistics to the given number of decimals.
    """

    def __init__(
        self,
        ground_truth_path: str,
        predictions_path: str,
        image_shape: Tuple[int, int] = (3840, 2160),
        confidence_threshold: Optional[float] = None,
        decimals: int = 3,
    ):
        self.decimals = decimals
        img_area = image_shape[0] * image_shape[1]
        if ground_truth_path.endswith(".json"):
            self.gt_dataset = YoloLabelsDataset.from_yolo_validation_json(
                yolo_val_json=ground_truth_path,
                image_shape=image_shape,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.gt_dataset = YoloLabelsDataset(
                folder_path=ground_truth_path,
                image_area=img_area,
                confidence_threshold=confidence_threshold,
            )
        if predictions_path.endswith(".json"):
            self.pred_dataset = YoloLabelsDataset.from_yolo_validation_json(
                yolo_val_json=predictions_path,
                image_shape=image_shape,
                confidence_threshold=confidence_threshold,
            )
        else:
            self.pred_dataset = YoloLabelsDataset(
                folder_path=predictions_path,
                image_area=img_area,
                confidence_threshold=confidence_threshold,
            )
        self.gt_all = self._get_filename_set(self.gt_dataset)

    def collect_results_per_class_and_size(
        self,
        classes: Iterable[ObjectClass] = ObjectClass,
        single_size_only: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes a dict with statistics (precision, recall, false positive rate
        - fpr, false negative rate - fnr, true negative rate - tnr) for each
        target class and bounding box size.

        Parameters
        ----------
        classes: Iterable[ObjectClass] = ObjectClass
            Which classes to evaluate (default is all).
        single_size_only: bool = False
            Whether to differentiate bounding box sizes (small, medium, large)
            or simply provide overall scores.

        Returns
        -------
        Dictionary with results:

            {
                [object_class]_[size]: {
                    "precision": float,
                    "recall": float,
                    "fpr": float,
                    "fnr:": float,
                    "tnr": float,
                }
            }
        """
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
        """Extract filtered_labels from a YoloLabelsDataset and return the
        corresponding image names as a set."""
        labels = yolo_dataset.get_filtered_labels()
        return set(k for k, v in labels.items() if len(v) > 0)

    def _compute_stats(
        self, ground_truth: Set, predictions: Set, box_size_name: str
    ) -> Dict[str, float]:
        """Compute statistics for a given set of ground truth and prediction image names."""
        size_all = box_size_name == "all"
        P = len(ground_truth)
        N = len(self.gt_all - ground_truth)
        PP = len(predictions)

        tp = len(ground_truth & predictions)
        fp = len(predictions - ground_truth)
        tn = len((self.gt_all - ground_truth) & (self.gt_all - predictions))
        fn = len(ground_truth - predictions)

        precision = round(tp / PP, self.decimals) if PP > 0 else np.nan
        recall = round(tp / P, self.decimals) if P > 0 else np.nan
        fpr = round(fp / N, self.decimals)
        fnr = round(fn / P, self.decimals) if P > 0 else np.nan
        tnr = round(tn / N, self.decimals)

        return {
            "precision": precision if size_all else np.nan,
            "recall": recall,
            "fpr": fpr if size_all else np.nan,
            "fnr": fnr,
            "tnr": tnr if size_all else np.nan,
        }
