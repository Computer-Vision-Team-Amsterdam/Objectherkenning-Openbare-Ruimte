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
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
        job_process_time=get_job_process_time(
            job_process_time_settings,
            is_first_pipeline_step=False,
        ),
    )
