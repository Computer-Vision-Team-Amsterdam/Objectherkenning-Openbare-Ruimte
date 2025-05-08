# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (  # noqa: E402
    SignalHandler,
    get_databricks_environment,
    get_job_process_time,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (  # noqa E402
    GoldSignalNotificationsManager,
    SilverObjectsPerDayManager,
    SilverObjectsPerDayQuarantineManager,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_submit_to_signalen_step():
    sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    catalog = settings["catalog"]
    schema = settings["schema"]
    job_process_time = get_job_process_time(
        is_first_pipeline_step=False,
    )
    az_tenant_id = settings["azure_tenant_id"]
    db_host = settings["reference_database"]["host"]
    db_name = settings["reference_database"]["name"]
    object_classes = settings["job_config"]["object_classes"]["names"]
    send_limits = settings["job_config"]["object_classes"]["send_limit"]
    exclude_private_terrain_detections = settings["job_config"][
        "exclude_private_terrain_detections"
    ]
    annotate_detection_images = settings["job_config"]["annotate_detection_images"]

    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)
    signalHandler = SignalHandler(
        sparkSession=sparkSession,
        catalog=catalog,
        schema=schema,
        device_id=settings["device_id"],
        signalen_settings=settings["signalen"],
        az_tenant_id=az_tenant_id,
        db_host=db_host,
        db_name=db_name,
        object_classes=object_classes,
        permit_mapping=settings["job_config"]["object_classes"]["permit_mapping"],
    )

    top_scores_df = SilverObjectsPerDayManager.get_top_pending_records(
        exclude_private_terrain_detections,
        az_tenant_id,
        db_host,
        db_name,
        send_limits=send_limits,
    )

    if (not top_scores_df) or top_scores_df.count() == 0:
        print("No data found for creating notifications. Stopping execution.")
    else:
        successful_notifications, unsuccessful_notifications = (
            signalHandler.process_notifications(
                top_scores_df, annotate_detection_images
            )
        )

        if successful_notifications:
            modified_schema = (
                GoldSignalNotificationsManager.remove_fields_from_table_schema(
                    fields_to_remove={"id", "processed_at"},
                )
            )
            successful_df = sparkSession.createDataFrame(
                successful_notifications, schema=modified_schema
            )
            GoldSignalNotificationsManager.insert_data(df=successful_df)

        if unsuccessful_notifications:
            modified_schema = (
                SilverObjectsPerDayQuarantineManager.remove_fields_from_table_schema(
                    fields_to_remove={"id", "processed_at"},
                )
            )
            unsuccessful_df = sparkSession.createDataFrame(
                unsuccessful_notifications, schema=modified_schema
            )
            SilverObjectsPerDayQuarantineManager.insert_data(df=unsuccessful_df)

    SilverObjectsPerDayManager.update_status(job_process_time=job_process_time)


if __name__ == "__main__":
    run_submit_to_signalen_step()
