# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

from helpers.utils_signalen import SignalHandler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.post_processing_pipeline.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
)
from objectherkenning_openbare_ruimte.post_processing_pipeline.table_manager import (  # noqa: E402
    TableManager,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_submit_to_signalen_step(
    sparkSession,
    catalog,
    schema,
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
        client_id,
        client_secret_name,
        access_token_url,
        base_url,
    )

    tableManager = TableManager(spark=SparkSession, catalog=catalog, schema=schema)

    # Get top pending records
    top_scores_df = signalHandler.get_top_pending_records(
        table_name="silver_objects_per_day", limit=10
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

    # Save notifications
    signalHandler.save_notifications(
        successful_notifications, unsuccessful_notifications
    )

    tableManager.update_status(
        table_name="silver_objects_per_day", job_process_time=job_process_time
    )


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("SignalHandler").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_submit_to_signalen_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
        job_process_time="2024-07-30 13:00:00",
    )
