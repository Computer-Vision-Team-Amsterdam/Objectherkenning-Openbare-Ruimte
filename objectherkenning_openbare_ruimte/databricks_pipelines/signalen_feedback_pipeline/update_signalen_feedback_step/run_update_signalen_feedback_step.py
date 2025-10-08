import os
from datetime import datetime

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    SignalHandler,
    get_databricks_environment,
    get_job_process_time,
    setup_tables,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    BronzeSignalNotificationsFeedbackManager,
    GoldSignalNotificationsManager,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (
    load_settings,
)


def run_update_signalen_feedback_step(
    spark_session,
    catalog,
    schema,
    device_id,
    client_id,
    client_secret_name,
    access_token_url,
    base_url,
    job_process_time,
):

    setup_tables(spark_session=spark_session, catalog=catalog, schema=schema)
    signalHandler = SignalHandler(
        spark_session,
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
        id = entry[GoldSignalNotificationsManager.id_column]
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
        fields_to_remove={GoldSignalNotificationsManager.id_column, "processed_at"}
    )

    signalen_feedback_df = spark_session.createDataFrame(  # noqa: F821
        signalen_feedback_entries, schema=modified_schema
    )
    BronzeSignalNotificationsFeedbackManager.insert_data(df=signalen_feedback_df)
    GoldSignalNotificationsManager.update_status(
        job_process_time=job_process_time, exclude_ids=ids_of_not_updated_status
    )


def main():
    spark_session = SparkSession.builder.appName("SignalenFeedback").getOrCreate()
    databricks_environment = get_databricks_environment(spark_session)
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.dirname(os.getcwd())))
    )
    config_file_path = os.path.join(project_root, "config_databricks.yml")
    settings = load_settings(config_file_path)["databricks_pipelines"][
        f"{databricks_environment}"
    ]

    run_update_signalen_feedback_step(
        spark_session=spark_session,
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


if __name__ == "__main__":
    main()
