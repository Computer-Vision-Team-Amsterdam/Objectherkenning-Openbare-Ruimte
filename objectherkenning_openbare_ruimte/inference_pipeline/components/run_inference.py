import os
import sys

from azure.ai.ml.constants import AssetTypes
from mldesigner import Input, Output, command_component
from ultralytics import YOLO

sys.path.append("../../..")

from cvtoolkit.helpers.file_helpers import find_image_paths  # noqa: E402

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

    params = {
        "source": image_paths,
        "imgsz": detection_params.get("img_size", 640),
        "save": detection_params.get("save_img_flag", False),
        "save_txt": detection_params.get("save_txt_flag", False),
        "save_conf": detection_params.get("save_conf_flag", False),
        "conf": detection_params.get("conf", 0.25),
        "project": project_path,
    }

    if tracking_flag:
        tracker = settings["inference_pipeline"]["inputs"]["tracker"]
        tracker_path = os.path.join(model_weights, tracker)
        tracking_persist_flag = settings["inference_pipeline"]["tracking_params"][
            "tracking_persist_flag"
        ]
        tracking_classes = settings["inference_pipeline"]["tracking_params"][
            "tracking_classes"
        ]

        labels_dir = os.path.join(project_path, "labels_manual")
        os.makedirs(labels_dir, exist_ok=True)

        model = YOLO(model=pretrained_model_path, task="track")

        params["persist"] = tracking_persist_flag
        params["tracker"] = tracker_path

        results = model.track(**params)

        for result in results:
            image_path = result.path  # Path of the input image
            image_name = os.path.basename(image_path)
            label_file = os.path.join(
                labels_dir, f"{os.path.splitext(image_name)[0]}.txt"
            )

            with open(label_file, "w") as f:
                for box in result.boxes:
                    class_id = int(box.cls)
                    if class_id in tracking_classes:
                        bbox = box.xywhn.tolist()[0]
                        track_id = box.id if hasattr(box, "id") else None

                        x_center, y_center, width, height = (
                            float(bbox[0]),
                            float(bbox[1]),
                            float(bbox[2]),
                            float(bbox[3]),
                        )
                        bbox_str = (
                            f"{x_center:.6f} {y_center:.6f} {width:.6f} {height:.6f}"
                        )
                        conf_score = float(box.conf)
                        conf_score_str = f"{conf_score:.6f}"

                        if track_id is not None:
                            line = f"{class_id} {bbox_str} {conf_score_str} {int(track_id)}\n"
                        else:
                            line = f"{class_id} {bbox_str} {conf_score_str}\n"

                        f.write(line)

        print("Tracking complete. Results saved.")
    else:
        model = YOLO(model=pretrained_model_path, task="detect")

        results = model(**params)
