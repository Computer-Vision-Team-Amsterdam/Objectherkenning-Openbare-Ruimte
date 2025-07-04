# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import argparse  # noqa: E402
import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    parse_manual_run_arg_to_settings,
    parse_task_args_to_settings,
    setup_arg_parser,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.submit_to_signalen_step.components.submit_to_signalen_step import (  # noqa: E402
    SubmitToSignalenStep,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main(args: argparse.Namespace) -> None:
    """
    Setup and run SubmitToSignalenStep.
    """
    spark_session = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(spark_session)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    print("Parsing job parameters...")
    settings = parse_task_args_to_settings(settings, args)
    settings = parse_manual_run_arg_to_settings(settings, args)

    print("Running the following task configuration:")
    for stadsdeel in settings["job_config"]["active_task"].keys():
        stadsdeel_str = str(settings["job_config"]["active_task"][stadsdeel])
        print(f"  - {stadsdeel}: {stadsdeel_str}")
    if settings["job_config"]["date"] is not None:
        print(
            f"  - will only process pending detections for date {settings["job_config"]["date"]}"
        )
    if len(settings["job_config"]["skip_ids"]) > 0:
        id_str = ", ".join(map(str, settings["job_config"]["skip_ids"]))
        print(f"  - will skip detection IDs: [{id_str}]")
    print("\n")

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)

    submitToSignalenStep = SubmitToSignalenStep(
        spark_session=spark_session, catalog=catalog, schema=schema, settings=settings
    )
    submitToSignalenStep.run_submit_to_signalen_step()


if __name__ == "__main__":
    parser = setup_arg_parser(prog="run_submit_to_signalen_step.py")
    main(args=parser.parse_args())
