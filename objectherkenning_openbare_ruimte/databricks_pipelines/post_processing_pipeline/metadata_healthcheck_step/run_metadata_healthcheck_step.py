# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.table_manager import (  # noqa: E402
    TableManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.metadata_healthcheck_step.components.metadata_healthcheck import (  # noqa: E402
    MetadataHealthChecker,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):
    metadataHealthCheker = MetadataHealthChecker(
        sparkSession, catalog, schema, job_process_time
    )
    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)

    bronze_frame_metadata_df = metadataHealthCheker.load_bronze_metadata(
        table_name="bronze_frame_metadata"
    )
    valid_frame_metadata, invalid_frame_metadata = (
        metadataHealthCheker.process_frame_metadata(
            bronze_frame_metadata=bronze_frame_metadata_df
        )
    )
    tableManager.write_to_table(
        valid_frame_metadata, table_name="silver_frame_metadata"
    )
    tableManager.write_to_table(
        invalid_frame_metadata, table_name="silver_frame_metadata_quarantine"
    )
    tableManager.update_status(
        table_name="bronze_frame_metadata", job_process_time=job_process_time
    )

    bronze_detection_metadata_df = metadataHealthCheker.load_bronze_metadata(
        table_name="bronze_detection_metadata"
    )
    valid_detection_metadata, invalid_detection_metadata = (
        metadataHealthCheker.process_detection_metadata(bronze_detection_metadata_df)
    )
    tableManager.write_to_table(
        valid_detection_metadata, table_name="silver_detection_metadata"
    )
    tableManager.write_to_table(
        invalid_detection_metadata, table_name="silver_detection_metadata_quarantine"
    )
    tableManager.update_status(
        table_name="bronze_detection_metadata", job_process_time=job_process_time
    )


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("MetadataHealthChecker").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time=get_job_process_time(
            settings["databricks_pipelines"]["job_process_time"],
            is_first_pipeline_step=False,
        ),
    )
