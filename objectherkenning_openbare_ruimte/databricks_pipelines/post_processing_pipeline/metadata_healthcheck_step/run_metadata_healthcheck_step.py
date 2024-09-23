# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)

# from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.detections import (  # noqa: E402
#     BronzeDetectionMetadataManager,
# )
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.frames import (  # noqa: E402
    BronzeFrameMetadataManager,
)

# from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.detections import (  # noqa: E402
#     SilverDetectionMetadataManager,
#     SilverDetectionMetadataQuarantineManager,
# )
# from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.frames import (  # noqa: E402
#     SilverFrameMetadataManager,
#     SilverFrameMetadataQuarantineManager,
# )
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    setup_tables,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_metadata_healthcheck_step(sparkSession, catalog, schema, job_process_time):

    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)
    _ = BronzeFrameMetadataManager.filter_valid_metadata()
    # invalid_frame_metadata = BronzeFrameMetadataManager.filter_invalid_metadata()

    # SilverFrameMetadataManager.insert_data(df=valid_frame_metadata)
    # SilverFrameMetadataQuarantineManager.insert_data(df=invalid_frame_metadata)
    # BronzeFrameMetadataManager.update_status(job_process_time=job_process_time)

    # silverFrameMetadataDf = SilverFrameMetadataManager.load_pending_rows_from_table()
    # valid_detection_metadata_df = BronzeDetectionMetadataManager.filter_valid_metadata(
    #     silver_frame_metadata_df=silverFrameMetadataDf
    # )
    # silverFrameMetadataQuarantineDf = (
    #     SilverFrameMetadataQuarantineManager.load_pending_rows_from_table()
    # )

    # invalid_detection_metadata_df = (
    #     BronzeDetectionMetadataManager.filter_invalid_metadata(
    #         silver_frame_metadata_quarantine_df=silverFrameMetadataQuarantineDf
    #     )
    # )
    # SilverDetectionMetadataManager.insert_data(df=valid_detection_metadata_df)
    # SilverDetectionMetadataQuarantineManager.insert_data(
    #     df=invalid_detection_metadata_df
    # )

    # BronzeDetectionMetadataManager.update_status(job_process_time=job_process_time)


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

    run_metadata_healthcheck_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
