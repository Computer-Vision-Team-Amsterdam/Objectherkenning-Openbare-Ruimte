# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import argparse  # noqa: E402
import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    parse_task_args_to_settings,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.submit_to_signalen_step.components.submit_to_signalen_step import (  # noqa: E402
    SubmitToSignalenStep,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main(args: argparse.Namespace):
    sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    settings = parse_task_args_to_settings(settings, args)
    print("Will use the following active tasks:")
    for stadsdeel in settings["job_config"]["active_task"].keys():
        stadsdeel_str = str(settings["job_config"]["active_task"][stadsdeel])
        print(f"{stadsdeel}: {stadsdeel_str}")

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)

    submitToSignalenStep = SubmitToSignalenStep(
        sparkSession=sparkSession, catalog=catalog, schema=schema, settings=settings
    )
    submitToSignalenStep.run_submit_to_signalen_step()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="run_submit_to_signalen.py")
    parser.add_argument(
        "--stadsdelen", type=str, default="", help="\"['name1', 'name2', ...]\""
    )
    parser.add_argument(
        "--send_limits",
        type=str,
        default="",
        help='"[{2: x, 3: y, 4: z}, {2: x2, 3: y2, 4: z2}, ...]"',
    )
    main(parser.parse_args())
