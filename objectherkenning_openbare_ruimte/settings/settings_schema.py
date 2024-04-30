from typing import Any, Dict, List

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


class DataDeliveryPipelineSpec(SettingsSpecModel):
    images_path: str
    detections_path: str
    metadata_path: str


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


class DataSampling(SettingsSpecModel):
    inputs: Dict[str, str]
    outputs: Dict[str, str]
    n_frames: int
    sampling_weight: float
    decos_buffer: float


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
    azure_iot: AzureIoTSpec
    data_delivery_pipeline: DataDeliveryPipelineSpec
    aml_experiment_details: AMLExperimentDetailsSpec
    convert_dataset: ConvertDataset = None
    convert_annotations: ConvertAnnotations = None
    data_sampling: DataSampling
    logging: LoggingSpec = LoggingSpec()
