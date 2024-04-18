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


class ConvertDataset(SettingsSpecModel):
    face_width: int = 1024
    input_old_datastore: str = "annotations_conversion_old"
    output_new_datastore: str = "annotations_conversion_new"


class ConvertAnnotations(SettingsSpecModel):
    input_datastore_name: str = "annotations_conversion_old"
    output_datastore_name: str = "annotations_conversion_new"
    final_datastore_name: str = "converted-dataset-oor"
    categories_file: str = "categories.json"
    separate_labels: bool = False
    label_folder: str = None


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
    azure_iot: AzureIoTSpec = None
    convert_dataset: ConvertDataset = None
    convert_annotations: ConvertAnnotations = None
    logging: LoggingSpec = LoggingSpec()
