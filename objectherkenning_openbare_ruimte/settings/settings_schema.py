from typing import Any, Dict, List, Optional, Tuple, Union

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


class ConvertDataset(SettingsSpecModel):
    face_width: int = 1024
    input_old_datastore: str
    output_new_datastore: str


class ConvertAnnotations(SettingsSpecModel):
    input_datastore_name: str = "annotations_conversion_old"
    output_datastore_name: str = "annotations_conversion_new"
    final_datastore_name: str = "converted-dataset-oor"
    image_storage_account: str = "cvodataweupgwapeg4pyiw5e"
    categories_file: str = "categories.json"
    separate_labels: bool = False
    label_folder: str = None


class DataSampling(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    n_frames: int
    sampling_weight: float
    decos_radius: float


class DefisheyeParameters(SettingsSpecModel):
    camera_matrix: List[List[float]]
    distortion_params: List[List[float]]
    input_image_size: Tuple[int, int]


class DistortionCorrectionSpec(SettingsSpecModel):
    defisheye_params: DefisheyeParameters
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


class InferenceModelParameters(SettingsSpecModel):
    batch_size: int = 1
    img_size: int = 640
    conf: float = 0.5
    save_img_flag: bool = False
    save_txt_flag: bool = False
    save_conf_flag: bool = False


class InferencePipelineSpec(SettingsSpecModel):
    model_params: InferenceModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None
    target_classes: List[int] = None
    sensitive_classes: List[int] = []
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    output_image_size: Optional[Tuple[int, int]] = None
    save_detection_images: bool = False
    save_detection_labels: bool = True
    save_all_images: bool = False
    defisheye_flag: bool = False


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


class PerformanceEvaluationSpec(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    dataset_name: str = ""
    model_name: str
    ground_truth_image_shape: List[int]
    predictions_image_shape: List[int]
    prediction_labels_rel_path: str = "labels"
    splits: List[str]
    target_classes: List[int]
    sensitive_classes: List[int]
    target_classes_conf: Optional[float] = None
    sensitive_classes_conf: Optional[float] = None
    plot_pr_curves: bool = True


class TrainingModelParameters(SettingsSpecModel):
    img_size: int = 1024
    batch: Union[float, int] = -1
    epochs: int = 100
    patience: int = 25
    n_classes: int = 3
    cos_lr: bool = False
    dropout: float = 0.0
    seed: int = 0
    box: float = 7.5
    cls: float = 0.5
    dfl: float = 1.5
    name_classes: List[str] = ["person", "license plate", "container"]
    rect: bool = False


class TrainingPipelineSpec(SettingsSpecModel):
    model_parameters: TrainingModelParameters
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None


class SweepModelParameters(SettingsSpecModel):
    img_size: int = 1024
    batch: Union[float, int] = -1
    epochs: int = 100
    patience: int = 25
    n_classes: int = 3
    name_classes: List[str] = ["person", "license plate", "container"]
    rect: bool = False


class SweepPipelineSpec(SettingsSpecModel):
    model_parameters: SweepModelParameters
    sweep_trials: int = 1
    inputs: Dict[str, str] = None
    outputs: Dict[str, str] = None


class WandbSpec(SettingsSpecModel):
    api_key: str
    mode: str = "disabled"


class ObjectherkenningOpenbareRuimteSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    convert_dataset: ConvertDataset = None
    convert_annotations: ConvertAnnotations = None
    data_sampling: DataSampling
    distortion_correction: DistortionCorrectionSpec = None
    frame_extraction: FrameExtractionSpec = None
    inference_pipeline: InferencePipelineSpec = None
    logging: LoggingSpec = LoggingSpec()
    performance_evaluation: PerformanceEvaluationSpec = None
    training_pipeline: TrainingPipelineSpec = None
    sweep_pipeline: SweepPipelineSpec = None
    wandb: WandbSpec = None
