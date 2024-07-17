from typing import Any, Dict, List, Tuple

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None
    src_dir: str = None
    ai_instrumentation_key: str = None


class DistortionCorrectionParameters(SettingsSpecModel):
    camera_matrix: List[List[float]]
    distortion_params: List[List[float]]
    input_image_size: Tuple[int, int]


class InferenceModelParameters(SettingsSpecModel):
    batch_size: int = 1
    img_size: int = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class InferencePipelineSpec(SettingsSpecModel):
    detection_params: InferenceModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    target_classes: List[int]
    sensitive_classes: List[int]
    output_image_size: Tuple[int, int]
    save_detection_images: bool
    save_detection_labels: bool
    defisheye_flag: bool
    defisheye_params: DistortionCorrectionParameters


class LoggingSpec(SettingsSpecModel):
    loglevel_own: str = "INFO"
    own_packages: List[str] = [
        "__main__",
        "objectherkenning_openbare_ruimte",
    ]
    extra_loglevels: Dict[str, str] = {}
    basic_config: Dict[str, Any] = {
        "level": "WARNING",
        "format": "%(asctime)s|%(levelname)-8s|%(name)s|%(message)s",
        "datefmt": "%Y-%m-%d %H:%M:%S",
    }
    ai_instrumentation_key: str = ""


class ObjectherkenningOpenbareRuimteSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    logging: LoggingSpec = LoggingSpec()
    inference_pipeline: InferencePipelineSpec = None
