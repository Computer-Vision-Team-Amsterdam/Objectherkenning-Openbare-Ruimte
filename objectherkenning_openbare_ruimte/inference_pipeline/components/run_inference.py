import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component

sys.path.append("../../..")

from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

from objectherkenning_openbare_ruimte.inference_pipeline.source.data_inference import (  # noqa: E402
    DataInference,
)
from objectherkenning_openbare_ruimte.settings.settings import (  # noqa: E402
    ObjectherkenningOpenbareRuimteSettings,
)

config_path = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..", "config.yml")
)

ObjectherkenningOpenbareRuimteSettings.set_from_yaml(config_path)
settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
aml_experiment_settings = settings["aml_experiment_details"]


@command_component(
    name="run_inference",
    display_name="Run inference on YOLOv8 model.",
    environment=f"azureml:{aml_experiment_settings['env_name']}:{aml_experiment_settings['env_version']}",
    code="../../../",
    is_deterministic=False,
)
def run_inference(
    mounted_dataset: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    model_weights: Input(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
    project_path: Output(type=AssetTypes.URI_FOLDER),  # type: ignore # noqa: F821
):
    """
    Pipeline step to run inference on YOLOv8 model.
    """
    image_paths = find_image_paths(mounted_dataset)
    model_name = settings["inference_pipeline"]["inputs"]["model_name"]
    pretrained_model_path = os.path.join(model_weights, model_name)
    detection_params = settings["inference_pipeline"]["detection_params"]
    tracking_flag = settings["inference_pipeline"]["tracking_params"]["tracking_flag"]

    if tracking_flag:
        tracker = settings["inference_pipeline"]["inputs"]["tracker"]
        tracker_path = os.path.join(model_weights, tracker)
        tracking_persist_flag = settings["inference_pipeline"]["tracking_params"][
            "tracking_persist_flag"
        ]

        detection_params["persist"] = tracking_persist_flag
        detection_params["tracker"] = tracker_path

    inference_pipeline = DataInference(
        images_folder=image_paths,
        inference_folder=project_path,
        model_name=model_name,
        pretrained_model_path=pretrained_model_path,
        inference_params=detection_params,
        target_classes=[2, 3, 4],
        sensitive_classes=[0, 1],
        tracking_flag=tracking_flag,
    )

    inference_pipeline.run_pipeline()


# labels_dir = os.path.join(project_path, "labels_manual")
# os.makedirs(labels_dir, exist_ok=True)
#
# if tracking_flag:
#    tracker = settings["inference_pipeline"]["inputs"]["tracker"]
#    tracker_path = os.path.join(model_weights, tracker)
#    tracking_persist_flag = settings["inference_pipeline"]["tracking_params"][
#        "tracking_persist_flag"
#    ]
#    tracking_classes = settings["inference_pipeline"]["tracking_params"][
#        "tracking_classes"
#    ]
#
#    model = YOLO(model=pretrained_model_path, task="track")
#
#    params["persist"] = tracking_persist_flag
#    params["tracker"] = tracker_path
#    results = model.track(**params)
#    processing_tools.process_tracking_results(results, labels_dir, tracking_classes)
# else:
#    print('Tracking flag is set to "False". Running detection only.')
#    model = YOLO(model=pretrained_model_path, task="detect")
#    params["show_labels"] = False
#    sensitive_classes = [0,1]
#    target_classes = [2,3,4]
#    batch_size = 32
#    image_dir = os.path.join(project_path, "images_manual")
#    os.makedirs(image_dir, exist_ok=True)
#    results = processing_tools.process_batches(model, params, image_paths, batch_size, image_dir, labels_dir, sensitive_classes, target_classes)"""
