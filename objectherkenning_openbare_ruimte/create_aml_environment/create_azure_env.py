from aml_interface.aml_interface import AMLInterface

from objectherkenning_openbare_ruimte.settings.settings import (
    ObjectherkenningOpenbareRuimteSettings,
)


def main():
    """
    This file creates an AML environment.
    """
    settings = ObjectherkenningOpenbareRuimteSettings.get_settings()
    aml_interface = AMLInterface()
    aml_interface.create_aml_environment(
        env_name=settings["aml_experiment_details"]["env_name"],
        build_context_path="objectherkenning_openbare_ruimte/create_aml_environment",
        dockerfile_path="Dockerfile",
    )


if __name__ == "__main__":
    ObjectherkenningOpenbareRuimteSettings.set_from_yaml("config.yml")
    main()
