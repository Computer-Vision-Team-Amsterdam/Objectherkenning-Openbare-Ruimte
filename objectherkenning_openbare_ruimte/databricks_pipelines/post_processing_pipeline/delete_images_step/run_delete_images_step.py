# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import argparse  # noqa: E402
import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    parse_detection_date_arg_to_settings,
    setup_arg_parser,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.delete_images_step.components.delete_images_step import (  # noqa: E402
    DeleteImagesStep,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main(args: argparse.Namespace) -> None:
    spark_session = SparkSession.builder.appName("ImageDeletion").getOrCreate()
    databricks_environment = get_databricks_environment(spark_session)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    print("Parsing job parameters...")
    settings = parse_detection_date_arg_to_settings(settings, args)
    if settings["job_config"]["detection_date"] is not None:
        print(
            f"-> will only process images for date {settings['job_config']['detection_date']}."
        )
    else:
        print("-> Will process all images.")
    print("\n")

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)

    deleteImagesStep = DeleteImagesStep(spark_session, catalog, schema, settings)
    deleteImagesStep.run()


if __name__ == "__main__":
    parser = setup_arg_parser(prog="run_delete_images_step.py")
    main(args=parser.parse_args())
