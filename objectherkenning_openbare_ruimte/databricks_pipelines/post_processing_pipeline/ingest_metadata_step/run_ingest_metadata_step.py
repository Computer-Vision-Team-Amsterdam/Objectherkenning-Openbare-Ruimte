# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from delta.tables import DeltaTable  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.ingest_metadata_step.components.data_ingestion import (  # noqa: E402
    DataLoader,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_ingest_metadata_step(
    sparkSesssion,
    catalog,
    schema,
    root_source,
    device_id,
    ckpt_frames_relative_path,
    ckpt_detections_relative_path,
    job_process_time,
):
    dataLoader = DataLoader(
        sparkSesssion,
        catalog,
        schema,
        root_source,
        device_id,
        ckpt_frames_relative_path,
        ckpt_detections_relative_path,
        job_process_time,
    )
    dataLoader.ingest_frame_metadata()
    dataLoader.ingest_detection_metadata()
    dbutils.jobs.taskValues.set(  # type: ignore[name-defined] # noqa: F821
        key="job_process_time", value=dataLoader.job_process_time.isoformat()
    )

    # Cleanup temporary files
    dataLoader.cleanup_temp_files()


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataIngestion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    silver_objects_per_day = DeltaTable.forPath(
        sparkSession,
        "catalog@stcatalogdpcvprdweu01.dfs.core.windows.net/__unitystorage/schemas/b594d498-13f9-4672-b576-31b8c2a79450/tables/681f83fc-5a4c-45cd-be51-ca652de746ec",
    )
    silver_objects_per_day.restoreToVersion(11)

    silver_detection_metadata = DeltaTable.forPath(
        sparkSession,
        "catalog@stcatalogdpcvprdweu01.dfs.core.windows.net/__unitystorage/schemas/b594d498-13f9-4672-b576-31b8c2a79450/tables/7cb1d6d2-6900-47be-a2dc-c6c54ec1961c",
    )
    silver_detection_metadata.restoreToVersion(12)

    silver_frame_metadata = DeltaTable.forPath(
        sparkSession,
        "catalog@stcatalogdpcvprdweu01.dfs.core.windows.net/__unitystorage/schemas/b594d498-13f9-4672-b576-31b8c2a79450/tables/c3d9e380-b4b7-4184-b3c6-4e2d918847f9",
    )
    silver_frame_metadata.restoreToVersion(12)

    run_ingest_metadata_step(
        sparkSesssion=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        root_source=settings["storage_account_root_path"],
        device_id=settings["device_id"],
        ckpt_frames_relative_path=settings["ckpt_frames_relative_path"],
        ckpt_detections_relative_path=settings["ckpt_detections_relative_path"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=True,
        ),
    )
