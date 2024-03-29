from aml_interface.aml_interface import AMLInterface


def main():
    """
    This file creates an AML environment.
    """
    aml_interface = AMLInterface()
    aml_interface.create_aml_environment(
        env_name="oor-test-environment",
        build_context_path="objectherkenning_openbare_ruimte/create_aml_environment",
        dockerfile_path="oor-environment.Dockerfile",
    )


if __name__ == "__main__":
    main()
