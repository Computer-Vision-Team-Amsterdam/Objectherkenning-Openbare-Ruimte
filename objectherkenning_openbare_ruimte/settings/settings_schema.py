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


class AzureIoTSpec(SettingsSpecModel):
    hostname: str
    device_id: str
    shared_access_key: str


class ConvertDataset(SettingsSpecModel):
    face_width: int = 1024
    input_old_datastore: str
    output_new_datastore: str


class ConvertAnnotations(SettingsSpecModel):
    input_datastore_name: str = "annotations_conversion_old"
    output_datastore_name: str = "annotations_conversion_new"
    final_datastore_name: str = "converted-dataset-oor"
    categories_file: str = "categories.json"
    separate_labels: bool = False
    label_folder: str = None


class DataDeliveryPipelineSpec(SettingsSpecModel):
    detections_path: str
    metadata_path: str
    ml_model_id: str
    project_version: str
    sleep_time: int


class InferenceModelParameters(SettingsSpecModel):
    img_size: int = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class DefisheyeParameters(SettingsSpecModel):
    camera_matrix: List[List[float]]
    distortion_params: List[List[float]]
    input_image_size: Tuple[int, int]


class DetectionPipelineSpec(SettingsSpecModel):
    images_path: str
    detections_path: str
    model_name: str
    pretrained_model_path: str
    inference_params: InferenceModelParameters
    defisheye_flag: bool
    defisheye_params: DefisheyeParameters
    target_classes: List[int]
    sensitive_classes: List[int]
    input_image_size: Tuple[int, int]
    output_image_size: Tuple[int, int]
    sleep_time: int
    training_mode: bool
    training_mode_destination_path: str


class DataSampling(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    n_frames: int
    sampling_weight: float
    decos_radius: float


class DistortionCorrectionSpec(SettingsSpecModel):
    cx: float
    cy: float
    k1: float
    k2: float


class FrameExtractionSpec(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    log_dir: str = "logs"
    exclude_dirs: List[str] = []
    exclude_files: List[str] = []
    fps: float


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
    luna_logs_dir: str = ""


class TrainingModelParameters(SettingsSpecModel):
    img_size: int = 1024
    batch: int = -1
    epochs: int = 100
    patience: int = 25
    n_classes: int = 3
    name_classes: List[str] = ["person", "license plate", "container"]


class TrainingPipelineSpec(SettingsSpecModel):
    model_parameters: TrainingModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None


class ObjectherkenningOpenbareRuimteSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    azure_iot: AzureIoTSpec
    convert_dataset: ConvertDataset = None
    convert_annotations: ConvertAnnotations = None
    data_delivery_pipeline: DataDeliveryPipelineSpec
    detection_pipeline: DetectionPipelineSpec
    data_sampling: DataSampling
    distortion_correction: DistortionCorrectionSpec
    frame_extraction: FrameExtractionSpec
    logging: LoggingSpec = LoggingSpec()
    training_pipeline: TrainingPipelineSpec = None
