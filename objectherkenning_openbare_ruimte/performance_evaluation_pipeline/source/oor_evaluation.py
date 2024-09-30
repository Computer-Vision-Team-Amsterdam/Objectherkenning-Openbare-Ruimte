import logging
import os
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (
    ObjectClass,
    compute_fb_score,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_image_stats import (
    PerImageEvaluator,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.per_pixel_stats import (
    PerPixelEvaluator,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.plot_utils import (
    save_fscore_curve,
    save_pr_curve,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.run_custom_coco_eval import (
    run_custom_coco_eval,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.yolo_to_coco import (
    convert_yolo_dataset_to_coco_json,
    convert_yolo_predictions_to_coco_json,
)

logger = logging.getLogger("performance_evaluation")

DEFAULT_TARGET_CLASSES = [
    ObjectClass.container,
    ObjectClass.mobile_toilet,
    ObjectClass.scaffolding,
]
DEFAULT_SENSITIVE_CLASSES = [
    ObjectClass.person,
    ObjectClass.license_plate,
]


class OOREvaluator:
    """
    This class is used to run evaluation of a trained YOLO model based on ground
    truth annotations and model predictions.

    OOREvaluator supports three evaluation methods:

    * Total Blurred Area evaluation for sensitive classes. This tells us the
      percentage of bounding boxes that are covered by predictions.

    * Per Image evaluation. This tells us the precision and recall based on
      whole images, i.e. if a single image contains at least one annotation of a
      certain class, does it also contain at least one prediction.

    * Custom COCO evaluation. This is a COCO-style evaluation of overall and per
      class precision and recall, for different bounding box sizes and
      confidence thresholds.

    Results are returned as Dictionaries that can optionally be converted to
    DataFrames.

    Parameters
    ----------

    ground_truth_base_folder: str
        Location of ground truth dataset (root folder, is expected to contain
        `images/` and `labels/` subfolders).
    predictions_base_folder: str
        Location of predictions (root folder, is expected to contain `labels/`
        subfolder).
    output_folder: Optional[Union[str, None]] = None
        Location where output will be stored. If None, the
        predictions_base_folder will be used.
    ground_truth_image_shape: Tuple[int, int] = (3840, 2160)
        Shape of ground truth images as (w, h).
    predictions_image_shape: Tuple[int, int] = (3840, 2160)
        Shape of prediction images as (w, h).
    dataset_name: str = ""
        Name of dataset, used in results plots.
    model_name: Optional[Union[str, None]] = None
        Name of the model used in the results. If no name is provided, the name
        of the predictions folder is used.
    gt_annotations_rel_path: str = "labels"
        Name of folder containing ground truth labels.
    pred_annotations_rel_path: str = "labels"
        Name of the folder containing prediction labels.
    splits: Union[List[str], None] = ["train", "val", "test"]
        Which splits to evaluate. Set to `None` of the data contains no splits.
    target_classes: Iterable[ObjectClass] = DEFAULT_TARGET_CLASSES
        Which object classes should be evaluated (default is ["container",
        "mobile_toilet", "scaffolding"]).
    sensitive_classes: Iterable[ObjectClass] = DEFAULT_SENSITIVE_CLASSES
        Which object classes should be treated as sensitive for the Total
        Blurred Area computation (default is ["person", "license_plate"]).
    target_classes_conf: Optional[float] = None
        Confidence threshold used for target classes. If not specified, all
        predictions will be evaluated.
    sensitive_classes_conf: Optional[float] = None
        Confidence threshold used for sensitive classes. If not specified, all
        predictions will be evaluated.
    single_size_only: bool = False
        Set to true to disable differentiation in bounding box sizes. Default is
        to evaluate for the sizes S, M, and L.
    """

    def __init__(
        self,
        ground_truth_base_folder: str,
        predictions_base_folder: str,
        output_folder: Optional[Union[str, None]] = None,
        ground_truth_image_shape: Tuple[int, int] = (3840, 2160),
        predictions_image_shape: Tuple[int, int] = (3840, 2160),
        dataset_name: str = "",
        model_name: Optional[Union[str, None]] = None,
        gt_annotations_rel_path: str = "labels",
        pred_annotations_rel_path: str = "labels",
        splits: Union[List[str], None] = ["train", "val", "test"],
        target_classes: List[ObjectClass] = DEFAULT_TARGET_CLASSES,
        sensitive_classes: List[ObjectClass] = DEFAULT_SENSITIVE_CLASSES,
        target_classes_conf: Optional[float] = None,
        sensitive_classes_conf: Optional[float] = None,
        single_size_only: bool = False,
    ):
        self.ground_truth_base_folder = ground_truth_base_folder
        self.predictions_base_folder = predictions_base_folder
        self.output_folder = output_folder
        self.ground_truth_image_shape = ground_truth_image_shape
        self.predictions_image_shape = predictions_image_shape
        self.dataset_name = dataset_name
        self.model_name = (
            model_name
            if model_name
            else os.path.basename(os.path.dirname(predictions_base_folder))
        )
        self.gt_annotations_rel_path = gt_annotations_rel_path
        self.pred_annotations_rel_path = pred_annotations_rel_path
        self.splits = splits if splits else [""]

        self.target_classes = target_classes
        self.sensitive_classes = sensitive_classes
        self.all_classes = self.target_classes + self.sensitive_classes

        self.target_classes_conf = target_classes_conf
        self.sensitive_classes_conf = sensitive_classes_conf
        self.single_size_only = single_size_only

        self._log_stats()

    def _log_stats(self) -> None:
        """Log number of annotation files in ground truth and prediction folders
        as sanity check."""
        for split in self.splits:
            split_name = split if split != "" else "all"
            gt_folder, pred_folder = self._get_folders_for_split(split)
            gt_count = len(
                [name for name in os.listdir(gt_folder) if name.endswith(".txt")]
            )
            pred_count = len(
                [name for name in os.listdir(pred_folder) if name.endswith(".txt")]
            )
            logger.info(
                f"Split: {split_name}, ground truth labels: {gt_count}, prediction labels: {pred_count}"
            )

    def _get_folders_for_split(self, split: str) -> Tuple[str, str]:
        """Generate the full path to ground truth and prediction annotation
        folders for a specific split."""
        ground_truth_folder = os.path.join(
            self.ground_truth_base_folder, self.gt_annotations_rel_path, split
        )
        prediction_folder = os.path.join(
            self.predictions_base_folder, self.pred_annotations_rel_path, split
        )
        return ground_truth_folder, prediction_folder

    def evaluate_tba(
        self,
        upper_half: bool = False,
        confidence_threshold: Optional[float] = None,
        single_size_only: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run Total Blurred Area evaluation for the sensitive classes. This tells
        us the percentage of bounding boxes that are covered by predictions.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]_[size]: {
                        "true_positives": float,
                        "false_positives": float,
                        "true_negatives": float,
                        "false_negatives:": float,
                        "precision": float,
                        "recall": float,
                        "f1_score": float,
                    }
                }
            }

        Parameters
        ----------
        upper_half: bool = False
            Whether to only consider the upper half of bounding boxes (relevant
            for people, to make sure the face is blurred).
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.
        single_size_only: Optional[bool] = None
            Optional: set to true to disable differentiation in bounding box
            sizes. If omitted, the initial confidence threshold at construction
            will be used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.sensitive_classes_conf
        if not single_size_only:
            single_size_only = self.single_size_only

        tba_results = dict()
        for split in self.splits:
            logger.info(
                f"Running TBA evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
            evaluator = PerPixelEvaluator(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                image_shape=self.predictions_image_shape,
                confidence_threshold=confidence_threshold,
                upper_half=upper_half,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            tba_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.sensitive_classes,
                single_size_only=single_size_only,
            )
        return tba_results

    def evaluate_per_image(
        self,
        confidence_threshold: Optional[float] = None,
        single_size_only: Optional[bool] = None,
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run Per Image evaluation for the sensitive classes. This tells us the
        precision and recall based on whole images, i.e. if a single image
        contains at least one annotation of a certain class, does it also
        contain at least one prediction.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]_[size]: {
                        "precision": float,
                        "recall": float,
                        "fpr": float,
                        "fnr": float,
                        "tnr": float,
                    }
                }
            }

        Parameters
        ----------
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.
        single_size_only: Optional[bool] = None
            Optional: set to true to disable differentiation in bounding box
            sizes. If omitted, the initial confidence threshold at construction
            will be used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.target_classes_conf
        if not single_size_only:
            single_size_only = self.single_size_only

        per_image_results = dict()
        for split in self.splits:
            logger.info(
                f"Running per-image evaluation for {self.model_name} / {split if split != '' else 'all'}"
            )
            ground_truth_folder, prediction_folder = self._get_folders_for_split(split)
            evaluator = PerImageEvaluator(
                ground_truth_path=ground_truth_folder,
                predictions_path=prediction_folder,
                image_shape=self.predictions_image_shape,
                confidence_threshold=confidence_threshold,
            )
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            per_image_results[key] = evaluator.collect_results_per_class_and_size(
                classes=self.target_classes, single_size_only=single_size_only
            )
        return per_image_results

    def evaluate_coco(
        self, confidence_threshold: Optional[float] = None
    ) -> Dict[str, Dict[str, Dict[str, float]]]:
        """
        Run custom COCO evaluation. This is a COCO-style evaluation of overall
        and per class precision and recall, for different bounding box sizes and
        confidence thresholds.

        TODO: come up with a way to incorporate the confidence threshold.

        The results are summarized in a dictionary as follows:

            {
                [model_name]_[split]: {
                    [object_class]: {
                        "AP@50-95_all": float,
                        "AP@75_all": float,
                        "AP@50_all": float,
                        "AP@50_small": float,
                        "AP@50_medium": float,
                        "AP@50_large": float,
                        "AR@50-95_all": float,
                        "AR@75_all": float,
                        "AR@50_all": float,
                        "AR@50_small": float,
                        "AR@50_medium": float,
                        "AR@50_large": float,
                    }
                }
            }

        Parameters
        ----------
        confidence_threshold: Optional[float] = None
            Optional: confidence threshold at which to compute statistics. If
            omitted, the initial confidence threshold at construction will be
            used.

        Returns
        -------
        Results as Dict[str, Dict[str, Dict[str, float]]] as described above.
        """
        if not confidence_threshold:
            confidence_threshold = self.target_classes_conf

        custom_coco_result: Dict[str, Dict[str, Dict[str, float]]] = dict()
        coco_eval_classes = {"all": self.all_classes}
        for obj_cls in self.all_classes:
            coco_eval_classes[obj_cls.name] = [obj_cls]

        # The custom COCO evaluation needs annotations in COCO JSON format, so we need to convert.
        ## Set output folders for COCO JSON files.
        if not self.output_folder:
            gt_output_dir = self.ground_truth_base_folder
        else:
            gt_output_dir = self.output_folder
        if not self.output_folder:
            pred_output_dir = self.predictions_base_folder
        else:
            pred_output_dir = self.output_folder
        ## Run conversion.
        gt_json_files = convert_yolo_dataset_to_coco_json(
            dataset_dir=self.ground_truth_base_folder,
            splits=self.splits,
            output_dir=gt_output_dir,
        )
        pred_json_files = convert_yolo_predictions_to_coco_json(
            predictions_dir=self.predictions_base_folder,
            image_shape=self.ground_truth_image_shape,
            labels_rel_path=self.pred_annotations_rel_path,
            splits=self.splits,
            output_dir=pred_output_dir,
            conf=(confidence_threshold if confidence_threshold else 0.0),
        )

        # Run evaluation
        for i, split in enumerate(self.splits):
            key = f"{self.model_name}_{split if split != '' else 'all'}"
            custom_coco_result[key] = dict()

            for target_cls_name, target_cls in coco_eval_classes.items():
                logger.info(
                    f"Running custom COCO evaluation for {self.model_name} / {split if split != '' else 'all'} / {target_cls_name}"
                )
                eval = run_custom_coco_eval(
                    coco_ground_truth_json=gt_json_files[i],
                    coco_predictions_json=pred_json_files[i],
                    predicted_img_shape=self.ground_truth_image_shape,
                    classes=target_cls,
                    print_summary=False,
                )
                subkey = target_cls_name
                custom_coco_result[key][subkey] = eval

        # Remove temporary JSON files
        for gt_file, pred_file in zip(gt_json_files, pred_json_files):
            os.remove(gt_file)
            os.remove(pred_file)

        return custom_coco_result

    def save_tba_results_to_csv(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Save TBA results dict as CSV file."""
        filename = os.path.join(self.output_folder, f"{self.model_name}-tba-eval.csv")
        _df_to_csv(tba_result_to_df(results), filename)

    def save_per_image_results_to_csv(
        self, results: Dict[str, Dict[str, Dict[str, float]]]
    ):
        """Save per-image results dict as CSV file."""
        filename = os.path.join(
            self.output_folder, f"{self.model_name}-per-image-eval.csv"
        )
        _df_to_csv(per_image_result_to_df(results), filename)

    def save_coco_results_to_csv(self, results: Dict[str, Dict[str, Dict[str, float]]]):
        """Save COCO results dict as CSV file."""
        filename = os.path.join(
            self.output_folder, f"{self.model_name}-custom-coco-eval.csv"
        )
        _df_to_csv(custom_coco_result_to_df(results), filename)

    def _compute_pr_curve_data(self, eval_func: Callable) -> pd.DataFrame:
        """
        Compute data needed to plot precision and recall curves. This method
        calls either `self.evaluate_tba` or `self.evaluate_per_image` with
        argument `single_size_only=True` for a range of confidence thresholds
        and returns the results in a DataFrame.

        Parameters
        ----------
        eval_func: Callable
            Either `self.evaluate_tba` or `self.evaluate_per_image`.

        Returns
        -------
        A pandas DataFrame with the evaluation results
        """
        confs = np.arange(0.05, 1.0, 0.05)
        dfs = []

        for conf in confs:
            tba_results = eval_func(confidence_threshold=conf, single_size_only=True)
            df = tba_result_to_df(tba_results)
            df.insert(4, "Conf", conf)
            df["F1"] = compute_fb_score(df["Precision"], df["Recall"], 1.0)
            df["F0.5"] = compute_fb_score(df["Precision"], df["Recall"], 0.5)
            df["F2"] = compute_fb_score(df["Precision"], df["Recall"], 2.0)
            dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def _plot_pr_f_curves(
        self,
        pr_df: pd.DataFrame,
        result_type: str,
        eval_classes: List[ObjectClass],
        output_dir: str = "",
        show_plot: bool = False,
    ):
        """Plot the precision and recall curves, and F-score curves."""
        for split, eval_class in product(self.splits, eval_classes):
            save_pr_curve(
                results_df=pr_df,
                dataset=self.dataset_name,
                split=split,
                target_class=eval_class,
                model_name=self.model_name,
                result_type=result_type,
                output_dir=output_dir,
                show_plot=show_plot,
            )
            save_fscore_curve(
                results_df=pr_df,
                dataset=self.dataset_name,
                split=split,
                target_class=eval_class,
                model_name=self.model_name,
                result_type=result_type,
                output_dir=output_dir,
                show_plot=show_plot,
            )

    def plot_tba_pr_f_curves(self, show_plot: bool = False):
        """
        Plot and save precision and recall curves and f-score curves for the
        total blurred area statistic. This will call evaluate_tba() for a each
        split and target_class for a range of confidence thresholds, which can
        take some time to compute.

        Parameters
        ----------
        show_plot: bool = False
            Whether or not to show the plot (True) or only save the image
            (False).
        """
        logger.info(f"Plotting TBA precision/recall curves for {self.model_name}")
        pr_curve_df = self._compute_pr_curve_data(self.evaluate_tba)
        self._plot_pr_f_curves(
            pr_df=pr_curve_df,
            result_type="total blurred area",
            eval_classes=self.sensitive_classes,
            output_dir=self.output_folder,
            show_plot=show_plot,
        )

    def plot_per_image_pr_f_curves(self, show_plot: bool = False):
        """
        Plot and save precision and recall curves and f-score curves for the
        per-image statistic. This will call evaluate_per-image() for a each
        split and target_class for a range of confidence thresholds, which can
        take some time to compute.

        Parameters
        ----------
        show_plot: bool = False
            Whether or not to show the plot (True) or only save the image
            (False).
        """
        logger.info(f"Plotting per-image precision/recall curves for {self.model_name}")
        pr_curve_df = self._compute_pr_curve_data(self.evaluate_per_image)
        self._plot_pr_f_curves(
            pr_df=pr_curve_df,
            result_type="per image",
            eval_classes=self.target_classes,
            output_dir=self.output_folder,
            show_plot=show_plot,
        )


def _default_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Convert TBA or PerImage results dictionary to Pandas DataFrame.
    """

    def _stat_to_header(stat: str) -> str:
        """For nicer column headings we transform 'true_positives' -> 'True Positives' etc."""
        if stat in ("fpr", "fnr", "tpr", "tnr"):
            return stat.upper()
        else:
            parts = [p.capitalize() for p in stat.split(sep="_")]
            return " ".join(parts)

    models = list(results.keys())
    categories = list(results[models[0]].keys())
    statistics = list(results[models[0]][categories[0]].keys())
    header = [
        "Model",
        "Split",
        "Object Class",
        "Size",
    ]
    header.extend([_stat_to_header(stat) for stat in statistics])

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            (cat_name, size) = cat.rsplit(sep="_", maxsplit=1)
            data: List[Any] = [model_name, split, cat_name, size]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def tba_result_to_df(results: Dict[str, Dict[str, Dict[str, float]]]) -> pd.DataFrame:
    """
    Convert TBA results dictionary to Pandas DataFrame.
    """
    return _default_result_to_df(results=results)


def per_image_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Convert Per Image results dictionary to Pandas DataFrame.
    """
    return _default_result_to_df(results=results)


def custom_coco_result_to_df(
    results: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Convert custom COCO results dictionary to Pandas DataFrame.
    """
    models = list(results.keys())
    categories = list(results[models[0]].keys())
    header = ["Model", "Split", "Object Class"]
    header.extend(list(results[models[0]][categories[0]].keys()))

    df = pd.DataFrame(columns=header)

    for model in models:
        (model_name, split) = model.rsplit(sep="_", maxsplit=1)
        for cat in categories:
            data: List[Any] = [model_name, split, cat]
            data.extend([val for val in results[model][cat].values()])
            df.loc[len(df)] = data

    return df


def _df_to_csv(df: pd.DataFrame, output_file: str):
    """Convenience method, currently not very useful but allows to change
    formatting of all CSVs in one place."""
    df.to_csv(output_file)
