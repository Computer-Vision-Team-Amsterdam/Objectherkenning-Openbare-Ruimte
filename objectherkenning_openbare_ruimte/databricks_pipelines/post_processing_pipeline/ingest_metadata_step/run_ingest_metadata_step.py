import os

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    get_databricks_environment,
    get_job_process_time,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.ingest_metadata_step.components.data_ingestion import (
    DataLoader,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (
    load_settings,
)


def run_ingest_metadata_step():
    spark_session = SparkSession.builder.appName("DataIngestion").getOrCreate()
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    databricks_environment = get_databricks_environment(spark_session)
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    catalog = settings["catalog"]
    schema = settings["schema"]
    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)

    dataLoader = DataLoader(
        spark_session=spark_session,
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
    dataLoader.ingest_json_metadata()
    dbutils.jobs.taskValues.set(  # type: ignore[name-defined] # noqa: F821
        key="job_process_time", value=dataLoader.job_process_time.isoformat()
    )

    # Cleanup temporary files
    dataLoader.cleanup_temp_files()


if __name__ == "__main__":
    run_ingest_metadata_step()
