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

    tableManager = TableManager(spark=sparkSession, catalog=catalog, schema=schema)

    # Get top pending records
    top_scores_df = signalHandler.get_top_pending_records(
        table_name="silver_objects_per_day", limit=20
    )

    # Check if there are records to process
    if top_scores_df.count() == 0:
        print("04: No data found for creating notifications. Stopping execution.")
        return

    print(
        f"04: Loaded {top_scores_df.count()} rows with top 10 scores from {signalHandler.catalog_name}.oor.silver_objects_per_day."
    )

    # Process notifications
    successful_notifications, unsuccessful_notifications = (
        signalHandler.process_notifications(top_scores_df)
    )

    if successful_notifications:
        modified_schema = tableManager.remove_fields_from_table_schema(
            table_name="gold_signal_notifications",
            fields_to_remove={"id", "processed_at"},
        )
        successful_df = sparkSession.createDataFrame(
            successful_notifications, schema=modified_schema
        )
        tableManager.write_to_table(
            df=successful_df, table_name="gold_signal_notifications"
        )

    if unsuccessful_notifications:
        modified_schema = tableManager.remove_fields_from_table_schema(
            table_name="silver_objects_per_day_quarantine",
            fields_to_remove={"id", "processed_at"},
        )
        unsuccessful_df = sparkSession.createDataFrame(
            unsuccessful_notifications, schema=modified_schema
        )
        tableManager.write_to_table(
            df=unsuccessful_df, table_name="silver_objects_per_day_quarantine"
        )

    tableManager.update_status(
        table_name="silver_objects_per_day", job_process_time=job_process_time
    )


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    job_process_time_settings = load_settings(config_file_path)["databricks_pipelines"][
        "job_process_time"
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
            job_process_time_settings,
            is_first_pipeline_step=False,
        ),
    )
