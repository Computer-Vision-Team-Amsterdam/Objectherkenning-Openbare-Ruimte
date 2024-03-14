from cvtoolkit.settings.settings_helper import GenericSettings, Settings
from pydantic import BaseModel

from objectherkenning_openbare_ruimte.settings.settings_schema import (
    ObjectherkenningOpenbareRuimteSettingsSpec,
)


class ObjectherkenningOpenbareRuimteSettings(Settings):  # type: ignore
    @classmethod
    def set_from_yaml(
        cls, filename: str, spec: BaseModel = ObjectherkenningOpenbareRuimteSettingsSpec
    ) -> "GenericSettings":
        return super().set_from_yaml(filename, spec)
