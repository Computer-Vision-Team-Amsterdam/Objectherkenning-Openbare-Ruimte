import logging
import os
import sys

import pandas as pd
from aml_interface.azure_logging import AzureLoggingConfigurer
from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from objectherkenning_openbare_ruimte.performance_evaluation_pipeline.source.oor_evaluation import (  # noqa: E402
    OOREvaluator,
    custom_coco_result_to_df,
    per_image_result_to_df,
    tba_result_to_df,
)
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()

log_settings = settings["logging"]
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
    model_name = settings["performance_evaluation"]["model_name"]
    predictions_img_shape = settings["performance_evaluation"][
        "predictions_image_shape"
    ]
    prediction_labels_rel_path = settings["performance_evaluation"][
        "prediction_labels_rel_path"
    ]

    logger.info(f"Running performance evaluation for model: {model_name}")

    os.makedirs(output_dir, exist_ok=True)

    oor_eval = OOREvaluator(
        ground_truth_base_folder=ground_truth_base_dir,
        predictions_base_folder=predictions_base_dir,
        output_folder=output_dir,
        predictions_image_shape=predictions_img_shape,
        model_name=model_name,
        pred_annotations_rel_path=prediction_labels_rel_path,
    )

    # Total Blurred Area evaluation
    tba_results = oor_eval.evaluate_tba()
    filename = os.path.join(output_dir, f"{model_name}-tba-eval.csv")
    _df_to_csv(tba_result_to_df(tba_results), filename)

    # Per Image evaluation
    per_image_results = oor_eval.evaluate_per_image()
    filename = os.path.join(output_dir, f"{model_name}-per-image-eval.csv")
    _df_to_csv(per_image_result_to_df(per_image_results), filename)

    # Custom COCO evaluation
    coco_results = oor_eval.evaluate_coco()
    filename = os.path.join(output_dir, f"{model_name}-custom-coco-eval.csv")
    _df_to_csv(custom_coco_result_to_df(coco_results), filename)


def _df_to_csv(df: pd.DataFrame, output_file: str):
    """Convenience method, currently not very useful but allows to change
    formatting of all CSVs in one place."""
    df.to_csv(output_file)
