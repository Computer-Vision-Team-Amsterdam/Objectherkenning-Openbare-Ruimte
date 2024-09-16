# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.detections import (  # noqa: E402
    BronzeDetectionMetadata,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.frames import (  # noqa: E402
    BronzeFrameMetadata,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (  # noqa: E402
    SilverFrameMetadata,
    SilverFrameMetadataQuarantine,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (  # noqa: E402
    TableManager,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):
    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)

    bronze_frame_metadata = BronzeFrameMetadata(
        spark=sparkSession, catalog=catalog, schema=schema
    )
    valid_frame_metadata = bronze_frame_metadata.get_valid_metadata()
    invalid_frame_metadata = bronze_frame_metadata.get_invalid_metadata()

    tableManager.write_to_table(
        valid_frame_metadata, table_name="silver_frame_metadata"
    )
    tableManager.write_to_table(
        invalid_frame_metadata, table_name="silver_frame_metadata_quarantine"
    )
    tableManager.update_status(
        table_name="bronze_frame_metadata", job_process_time=job_process_time
    )

    bronze_detection_metadata = BronzeDetectionMetadata(
        spark=sparkSession, catalog=catalog, schema=schema
    )
    silver_frame_metadata = SilverFrameMetadata(
        spark=sparkSession, catalog=catalog, schema=schema
    )

    # make TableManager an abstract class and then use get_table_name below()
    silver_frame_metadata_pending = silver_frame_metadata.load_pending_rows_from_table(
        table_name="silver_frame_metadata"
    )
    valid_detection_metadata = bronze_detection_metadata.get_valid_metadata(
        silver_frame_metadata=silver_frame_metadata_pending
    )

    silver_frame_metadata_quarantine = SilverFrameMetadataQuarantine(
        spark=sparkSession, catalog=catalog, schema=schema
    )
    silver_frame_metadata_quarantine_pending = (
        silver_frame_metadata_quarantine.load_pending_rows_from_table(
            table_name="silver_frame_metadata_quarantine"
        )
    )
    invalid_detection_metadata = bronze_detection_metadata.get_invalid_metadata(
        silver_frame_metadata_quarantine=silver_frame_metadata_quarantine_pending
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
    config_file_path = os.path.join(project_root, "config_db.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    job_process_time_settings = load_settings(config_file_path)["databricks_pipelines"][
        "job_process_time"
    ]

    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
