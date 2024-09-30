import logging
import os
import sys

from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.metrics.metrics_utils import (  # noqa: E402
    ObjectClass,
)
from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.oor_evaluation import (  # noqa: E402
    OOREvaluator,
)
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

azure_logging_configurer = AzureLoggingConfigurer(settings["logging"])
azure_logging_configurer.setup_oor_logging()
logger = logging.getLogger("performance_evaluation")

aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="evaluate_model",
    display_name="Evaluate model predictions.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=True,
)
def evaluate_model(
    ground_truth_base_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    output_dir: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Run evaluation of a model based on ground truth annotations and model
    predictions.

    This pipeline runs three evaluation methods:

    * Total Blurred Area evaluation for sensitive classes. This tells us the
      percentage of bounding boxes that are covered by predictions.

    * Per Image evaluation. This tells us the precision and recall based on
      whole images, i.e. if a single image contains at least one annotation of a
      certain class, does it also contain at least one prediction.

    * Custom COCO evaluation. This is a COCO-style evaluation of overall and per
      class precision and recall, for different bounding box sizes and
      confidence thresholds.

    Results are stored as CSV files in the chosen output location.

    Parameters
    ----------

    ground_truth_base_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of ground truth dataset (root folder, is expected to contain
        `images/` and `labels/` subfolders).
    predictions_base_dir: Input(type=AssetTypes.URI_FOLDER)
        Location of predictions (root folder, is expected to contain `labels/`
        subfolder).
    output_dir: Output(type=AssetTypes.URI_FOLDER)
        Location where output will be stored.
    """
    eval_settings = settings["performance_evaluation"]
    dataset_name = eval_settings["dataset_name"]
    model_name = eval_settings["model_name"]
    ground_truth_img_shape = eval_settings["ground_truth_image_shape"]
    predictions_img_shape = eval_settings["predictions_image_shape"]
    prediction_labels_rel_path = eval_settings["prediction_labels_rel_path"]
    splits = eval_settings["splits"]
    target_classes = eval_settings["target_classes"]
    sensitive_classes = eval_settings["sensitive_classes"]
    target_classes_conf = eval_settings["target_classes_conf"]
    sensitive_classes_conf = eval_settings["sensitive_classes_conf"]

    logger.info(f"Running performance evaluation for model: {model_name}")

    os.makedirs(output_dir, exist_ok=True)

    oor_eval = OOREvaluator(
        ground_truth_base_folder=ground_truth_base_dir,
        predictions_base_folder=predictions_base_dir,
        output_folder=output_dir,
        ground_truth_image_shape=ground_truth_img_shape,
        predictions_image_shape=predictions_img_shape,
        dataset_name=dataset_name,
        model_name=model_name,
        pred_annotations_rel_path=prediction_labels_rel_path,
        splits=splits,
        target_classes=[ObjectClass(val) for val in target_classes],
        sensitive_classes=[ObjectClass(val) for val in sensitive_classes],
        target_classes_conf=target_classes_conf,
        sensitive_classes_conf=sensitive_classes_conf,
    )

    # Total Blurred Area evaluation
    tba_results = oor_eval.evaluate_tba()
    oor_eval.save_tba_results_to_csv(results=tba_results)

    # Per Image evaluation
    per_image_results = oor_eval.evaluate_per_image()
    oor_eval.save_per_image_results_to_csv(results=per_image_results)

    # Custom COCO evaluation
    coco_results = oor_eval.evaluate_coco()
    oor_eval.save_coco_results_to_csv(results=coco_results)

    # Plot precision/recall curves
    if eval_settings["plot_pr_curves"]:
        oor_eval.plot_tba_pr_f_curves(show_plot=False)
        oor_eval.plot_per_image_pr_f_curves(show_plot=False)
