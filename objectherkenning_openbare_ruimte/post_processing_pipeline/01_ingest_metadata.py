# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

from helpers.data_ingestion import DataLoader  # noqa: E402
from helpers.databricks_workspace import get_databricks_environment  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

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
        key="job_process_time", value=dataLoader.job_process_time
    )

    # Cleanup temporary files
    dataLoader.cleanup_temp_files()


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("DataIngestion").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_ingest_metadata_step(
        sparkSesssion=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        root_source=settings["storage_account_root_path"],
        device_id=settings["device_id"],
        ckpt_frames_relative_path=settings["ckpt_frames_relative_path"],
        ckpt_detections_relative_path=settings["ckpt_detections_relative_path"],
        job_process_time="2024-07-30 13:00:00",
    )
