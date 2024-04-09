from typing import Any, Dict, List

from pydantic import BaseModel


class SettingsSpecModel(BaseModel):
    class Config:
        extra = "forbid"


class AMLExperimentDetailsSpec(SettingsSpecModel):
    compute_name: str = None
    env_name: str = None
    env_version: int = None


class AzureIoTSpec(SettingsSpecModel):
    hostname: str
    device_id: str
    shared_access_key: str


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


class ObjectherkenningOpenbareRuimteSettingsSpec(SettingsSpecModel):
    class Config:
        extra = "forbid"

    customer: str
    aml_experiment_details: AMLExperimentDetailsSpec
    azure_iot: AzureIoTSpec
    distortion_correction: DistortionCorrectionSpec
    frame_extraction: FrameExtractionSpec
    logging: LoggingSpec = LoggingSpec()
