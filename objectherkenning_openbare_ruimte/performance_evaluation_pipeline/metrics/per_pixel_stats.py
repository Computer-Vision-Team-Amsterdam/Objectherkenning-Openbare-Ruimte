import logging
from typing import Dict, Iterable, Optional, Tuple

import numpy as np
import numpy.typing as npt
from cvtoolkit.datasets.yolo_labels_dataset import YoloLabelsDataset

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (
    BoxSize,
    ObjectClass,
    generate_binary_mask,
)

logger = logging.getLogger(__name__)


class PixelStats:
    """
    This class keeps track of per-pixel statistics for bounding box accuracy.
    """

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_statistics_based_on_masks(
        self, true_mask: np.ndarray, predicted_mask: np.ndarray
    ) -> None:
        """
        Updates statistics based on a pair of binary masks for an image, one
        containing the ground truth and one the predictions.

        Parameters
        ----------
        true_mask: np.ndarray
            Binary numpy array of shape (height, width) containing the ground
            truth annotation, with 'True' values for each pixel within an
            annotation bounding box.
        predicted_mask: np.ndarray
            Binary numpy array of shape (height, width) containing the
            predictions, with 'True' values for each pixel within an annotation
            bounding box.
        """
        self.tp += np.count_nonzero(np.logical_and(true_mask, predicted_mask))
        self.fp += np.count_nonzero(np.logical_and(~true_mask, predicted_mask))
        self.tn += np.count_nonzero(np.logical_and(~true_mask, ~predicted_mask))
        self.fn += np.count_nonzero(np.logical_and(true_mask, ~predicted_mask))

    def get_statistics(self, size_all: bool, decimals: int = 3) -> Dict[str, float]:
        """
        Return statistics after all masks have been added to the calculation.
        Computes precision, recall and f1_score. Also returns the total number
        of pixels that have been counted as True Positive, False Positive, True
        Negative, False Negative.

        Parameters
        ----------
        size_all: bool
            Indicates whether the statistics are computed over all bounding box
            sizes (True) or whether a subset is evaluated (False). If False,
            only True Positive, False Negative, and Recall are returned as the
            other statistics are not meaningful.
        decimals: int = 3
            Rounds precision, recall, and f1_score to the given number of decimals.

        Returns
        -------
        Dictionary with statistics:

            {
                "true_positives": int,
                "false_positives": int,
                "true_negatives": int,
                "false_negatives": int,
                "precision": float,
                "recall": float,
                "f1_score": float,
            }
        """
        precision = (
            round(self.tp / (self.tp + self.fp), decimals)
            if self.tp + self.fp > 0
            else np.nan
        )
        recall = (
            round(self.tp / (self.tp + self.fn), decimals)
            if self.tp + self.fn > 0
            else np.nan
        )
        f1_score = (
            round(2 * precision * recall / (precision + recall), decimals)
            if precision and recall
            else np.nan
        )

        return {
            "true_positives": self.tp,
            "false_positives": self.fp if size_all else np.nan,
            "true_negatives": self.tn if size_all else np.nan,
            "false_negatives": self.fn,
            "precision": precision if size_all else np.nan,
            "recall": recall,
            "f1_score": f1_score if size_all else np.nan,
        }


class PerPixelEvaluator:
    """
    This class is used to run per-pixel evaluation over a dataset of ground
    truth and prediction labels. For each object class and bounding box size
    (small, medium, large) it will compute precision, recall, and f1-score based
    on the per-pixel accuracy of the predictions.

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
        upper_half: bool = False
            Whether to only consider the upper half of bounding boxes (relevant
            for people, to make sure the face is blurred).
        decimals: int = 3
            Round statistics to the given number of decimals.
    """

    def __init__(
        self,
        ground_truth_path: str,
        predictions_path: str,
        image_shape: Tuple[int, int] = (3840, 2160),
        confidence_threshold: Optional[float] = None,
        upper_half: bool = False,
        decimals: int = 3,
    ):
        self.img_shape = image_shape
        self.upper_half = upper_half
        self.decimals = decimals
        img_area = self.img_shape[0] * self.img_shape[1]
        if not confidence_threshold:
            confidence_threshold = 0.0

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

    def _get_per_pixel_statistics(
        self,
        true_labels: Dict[str, npt.NDArray],
        predicted_labels: Dict[str, npt.NDArray],
        size_all: bool,
    ) -> Dict[str, float]:
        """
        Calculates per pixel statistics (tp, tn, fp, fn, precision, recall, f1
        score) for the annotations and predictions provided.

        Each key in the dict is an image, each value is a ndarray (n_detections, 5)
        The 5 columns are in the YOLO format, i.e. (target_class, x_c, y_c, width, height)

        Parameters
        ----------
        true_labels: Dict[str, npt.NDArray]
            Ground truth annotations.
        predicted_labels: Dict[str, npt.NDArray]
            Predictions.
        size_all: bool
            Indicates whether the statistics are computed over all bounding box
            sizes (True) or whether a subset is evaluated (False). If False,
            only True Positive, True Negative, and Precision are returned as the
            other statistics are not meaningful.

        Returns
        -------
        Dictionary with statistics:

            {
                "true_positives": int,
                "false_positives": int,
                "true_negatives": int,
                "false_negatives": int,
                "precision": float,
                "recall": float,
                "f1_score": float,
            }
        """
        pixel_stats = PixelStats()

        (img_width, img_height) = self.img_shape

        for image_id in true_labels.keys():
            tba_true_mask = generate_binary_mask(
                true_labels[image_id][:, 1:5],
                image_width=img_width,
                image_height=img_height,
                consider_upper_half=self.upper_half,
            )
            if image_id in predicted_labels.keys():
                pred_labels = predicted_labels[image_id][:, 1:5]
            else:
                pred_labels = np.array([])
            tba_pred_mask = generate_binary_mask(
                pred_labels,
                image_width=img_width,
                image_height=img_height,
                consider_upper_half=self.upper_half,
            )

            pixel_stats.update_statistics_based_on_masks(
                true_mask=tba_true_mask, predicted_mask=tba_pred_mask
            )

        results = pixel_stats.get_statistics(size_all=size_all, decimals=self.decimals)

        return results

    def collect_results_per_class_and_size(
        self,
        classes: Iterable[ObjectClass] = ObjectClass,
        single_size_only: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """
        Computes a dict with statistics (tn, tp, fp, fn, precision, recall, f1)
        for each target class and bounding box size.

        Parameters
        ----------
        classes: Iterable[ObjectClass] = ObjectClass,
            Which classes to evaluate (default is all).
        single_size_only: bool = False,
            Whether to differentiate bounding box sizes (small, medium, large)
            or simply provide overall scores.

        Returns
        -------
        Dictionary with results:

            {
                [object_class]_[size]: {
                    "true_positives": float,
                    "false_positives": float,
                    "true_negatives": float,
                    "false_negatives": float,
                    "precision": float,
                    "recall": float,
                    "f1_score": float,
                }
            }
        """
        results = {}

        for target_class in classes:
            self.pred_dataset.reset_filter()
            predicted_target_class = self.pred_dataset.filter_by_class(
                target_class.value
            ).get_filtered_labels()

            box_sizes = BoxSize.from_objectclass(target_class).to_dict(single_size_only)

            for box_size_name, box_size in box_sizes.items():
                size_all = box_size_name == "all"

                self.gt_dataset.reset_filter()
                true_target_class_size = (  # i.e. true_person_small
                    self.gt_dataset.filter_by_class(class_to_keep=target_class.value)
                    .filter_by_size_percentage(perc_to_keep=box_size)
                    .get_filtered_labels()
                )

                results[f"{target_class.name}_{box_size_name}"] = (
                    self._get_per_pixel_statistics(
                        true_labels=true_target_class_size,
                        predicted_labels=predicted_target_class,
                        size_all=size_all,
                    )
                )

        return results
