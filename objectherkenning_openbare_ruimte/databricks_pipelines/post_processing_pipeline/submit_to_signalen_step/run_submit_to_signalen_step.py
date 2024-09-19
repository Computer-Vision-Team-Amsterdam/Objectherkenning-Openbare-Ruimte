# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.gold.notifications import (  # noqa E402
    GoldSignalNotificationsManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.silver.objects import (  # noqa: E402
    SilverObjectsPerDayManager,
    SilverObjectsPerDayQuarantineManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils_signalen import (  # noqa: E402
    SignalHandler,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_submit_to_signalen_step(
    sparkSession,
    catalog,
    schema,
    device_id,
    client_id,
    client_secret_name,
    access_token_url,
    base_url,
    job_process_time,
):
    # Initialize SignalHandler
    signalHandler = SignalHandler(
        sparkSession,
        catalog,
        schema,
        device_id,
        client_id,
        client_secret_name,
        access_token_url,
        base_url,
    )

    silverObjectsPerDayManager = SilverObjectsPerDayManager(
        spark=sparkSession, catalog=catalog, schema=schema
    )
    top_scores_df = silverObjectsPerDayManager.get_top_pending_records(limit=20)

    if top_scores_df.count() == 0:
        print("No data found for creating notifications. Stopping execution.")
        return

    successful_notifications, unsuccessful_notifications = (
        signalHandler.process_notifications(top_scores_df)
    )

    goldSignalNotificationsManager = GoldSignalNotificationsManager(
        spark=sparkSession, catalog=catalog, schema=schema
    )
    silverObjectsPerDayQuarantineManager = SilverObjectsPerDayQuarantineManager(
        spark=sparkSession, catalog=catalog, schema=schema
    )

    if successful_notifications:
        modified_schema = (
            goldSignalNotificationsManager.remove_fields_from_table_schema(
                fields_to_remove={"id", "processed_at"},
            )
        )
        successful_df = sparkSession.createDataFrame(
            successful_notifications, schema=modified_schema
        )
        goldSignalNotificationsManager.insert_data(df=successful_df)

    if unsuccessful_notifications:
        modified_schema = (
            silverObjectsPerDayQuarantineManager.remove_fields_from_table_schema(
                fields_to_remove={"id", "processed_at"},
            )
        )
        unsuccessful_df = sparkSession.createDataFrame(
            unsuccessful_notifications, schema=modified_schema
        )
        silverObjectsPerDayQuarantineManager.insert_data(df=unsuccessful_df)

    silverObjectsPerDayManager.update_status(job_process_time=job_process_time)


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_db.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    run_submit_to_signalen_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=False,
        ),
    )
