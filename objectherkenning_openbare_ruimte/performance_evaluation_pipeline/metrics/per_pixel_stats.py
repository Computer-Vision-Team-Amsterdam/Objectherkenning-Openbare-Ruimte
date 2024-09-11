import logging
from typing import Dict, Iterable, Tuple

import numpy as np
import numpy.typing as npt
from cvtoolkit.datasets.yolo_labels_dataset import YoloLabelsDataset

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    BoxSize,
    ObjectClass,
    generate_binary_mask,
)

logger = logging.getLogger(__name__)


class PixelStats:

    def __init__(self):
        self.tp = 0
        self.fp = 0
        self.tn = 0
        self.fn = 0

    def update_statistics_based_on_masks(self, true_mask, predicted_mask):
        """
        Computes statistics for a given pair of binary masks.

        Parameters
        ----------
        true_mask numpy array of shape (height, width)
        predicted_mask numpy array of shape (height, width)

        Returns
        -------

        """
        self.tp += np.count_nonzero(np.logical_and(true_mask, predicted_mask))
        self.fp += np.count_nonzero(np.logical_and(~true_mask, predicted_mask))
        self.tn += np.count_nonzero(np.logical_and(~true_mask, ~predicted_mask))
        self.fn += np.count_nonzero(np.logical_and(true_mask, ~predicted_mask))

    def get_statistics(self, precision: int = 3):
        """
        Return statistics after all masks have been added to the calculation.

        Computes precision, recall and f1_score only in the end since it is redundant to
        do this intermediately.

        Returns
        -------

        """
        prec = (
            round(self.tp / (self.tp + self.fp), precision)
            if self.tp + self.fp > 0
            else None
        )
        recall = (
            round(self.tp / (self.tp + self.fn), precision)
            if self.tp + self.fn > 0
            else None
        )
        f1_score = (
            round(2 * prec * recall / (prec + recall), precision)
            if prec and recall
            else None
        )

        return {
            "true_positives": self.tp,
            "false_positives": self.fp,
            "true_negatives": self.tn,
            "false_negatives:": self.fn,
            "precision": prec,
            "recall": recall,
            "f1_score": f1_score,
        }


class EvaluatePixelWise:

    def __init__(
        self,
        ground_truth_path: str,
        predictions_path: str,
        image_shape: Tuple[int, int] = (3840, 2160),
        upper_half: bool = False,
        precision: int = 3,
    ):
        self.img_shape = image_shape
        self.upper_half = upper_half
        self.precision = precision
        img_area = self.img_shape[0] * self.img_shape[1]
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

    def _get_per_pixel_statistics(
        self,
        true_labels: Dict[str, npt.NDArray],
        predicted_labels: Dict[str, npt.NDArray],
    ):
        """
        Calculates per pixel statistics (tp, tn, fp, fn, precision, recall, f1 score)

        Each key in the dict is an image, each value is a ndarray (n_detections, 5)
        The 6 columns are in the yolo format, i.e. (target_class, x_c, y_c, width, height)

        Parameters
        ----------
        true_labels
        predicted_labels

        Returns
        -------

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

        results = pixel_stats.get_statistics(self.precision)

        return results

    def collect_results_per_class_and_size(
        self,
        classes: Iterable[ObjectClass] = ObjectClass,
        single_size_only: bool = False,
    ) -> Dict[str, Dict[str, float]]:
        """

        Computes a dict with statistics (tn, tp, fp, fn, precision, recall, f1) for each target class and size.

        Parameters
        ----------


        Returns:
        -------

        """
        results = {}

        for target_class in classes:
            self.pred_dataset.reset_filter()
            predicted_target_class = self.pred_dataset.filter_by_class(
                target_class.value
            ).get_filtered_labels()

            box_sizes = BoxSize.from_objectclass(target_class).to_dict(single_size_only)

            for box_size_name, box_size in box_sizes.items():
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
                    )
                )

        return results