import argparse
import os

os.environ.pop(
    "OPENSSL_FORCE_FIPS_MODE", None
)  # TEMPORARY FIX, see https://github.com/opencv/opencv-python/issues/1184

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    get_databricks_environment,
    parse_detection_date_arg_to_settings,
    parse_manual_run_arg_to_settings,
    setup_arg_parser,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.data_enrichment_step.components.data_enrichment import (  # noqa: E402
    DataEnrichment,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def main(args: argparse.Namespace) -> None:
    spark_session = SparkSession.builder.appName("DataEnrichment").getOrCreate()
    databricks_environment = get_databricks_environment(spark_session)

    # Prevent user warnings in vulnerable bridge handler when converting between spark and pandas
    spark_session.conf.set("spark.sql.execution.arrow.pyspark.enabled", "false")

    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    print("Parsing job parameters...")
    settings = parse_detection_date_arg_to_settings(settings, args)
    settings = parse_manual_run_arg_to_settings(settings, args)

    if settings["job_config"]["detection_date"] is not None:
        print(
            f"  - will only process pending detections for date {settings['job_config']['detection_date']}"
        )

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)

    data_enrichment = DataEnrichment(
        spark_session=spark_session, catalog=catalog, schema=schema, settings=settings
    )
    data_enrichment.run_data_enrichment_step()


if __name__ == "__main__":
    parser = setup_arg_parser(prog="run_data_enrichment_step.py")
    main(args=parser.parse_args())
