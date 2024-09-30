# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from datetime import datetime  # noqa: E402

from pyspark.sql import SparkSession  # noqa: E402

from objectherkenning_openbare_ruimte.databricks_pipelines.common.databricks_workspace import (  # noqa: E402
    get_databricks_environment,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.bronze.feedback import (  # noqa: E402
    BronzeSignalNotificationsFeedbackManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.gold.notifications import (  # noqa: E402
    GoldSignalNotificationsManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils import (  # noqa: E402
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.utils_signalen import (  # noqa: E402
    SignalHandler,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_update_signalen_feedback_step(
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

    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)
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

    signalen_feedback_entries = []
    ids_of_not_updated_status = []
    for (
        entry
    ) in GoldSignalNotificationsManager.load_pending_rows_from_table().collect():
        id = entry["id"]
        signal_status = signalHandler.get_signal(sig_id=entry["signal_id"])["status"]
        if signal_status["state_display"] != "Gemeld":
            status, text, user, status_update_time = (
                signal_status["state_display"],
                signal_status["text"],
                signal_status["user"],
                signal_status["created_at"],
            )
            # Convert the date string to a datetime object
            date_obj = datetime.fromisoformat(status_update_time)

            formatted_date = datetime(
                year=date_obj.year,
                month=date_obj.month,
                day=date_obj.day,
                hour=date_obj.hour,
                minute=date_obj.minute,
                second=date_obj.second,
            )
            table_entry = (entry["signal_id"], status, text, user, formatted_date)
            signalen_feedback_entries.append(table_entry)
        else:
            ids_of_not_updated_status.append(id)

    modified_schema = GoldSignalNotificationsManager.remove_fields_from_table_schema(
        fields_to_remove={"id", "processed_at"}
    )

    signalen_feedback_df = sparkSession.createDataFrame(  # noqa: F821
        signalen_feedback_entries, schema=modified_schema
    )
    BronzeSignalNotificationsFeedbackManager.insert_data(df=signalen_feedback_df)
    GoldSignalNotificationsManager.update_status(
        job_process_time=job_process_time, exclude_ids=ids_of_not_updated_status
    )


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("SignalenFeedback").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    run_update_signalen_feedback_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        device_id=settings["device_id"],
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
        job_process_time=get_job_process_time(
            is_first_pipeline_step=True,
        ),
    )
