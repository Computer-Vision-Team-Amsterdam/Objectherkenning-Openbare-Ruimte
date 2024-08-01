# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821
import sys  # noqa: E402

sys.path.append("..")

from datetime import datetime  # noqa: E402

from helpers.databricks_workspace import get_databricks_environment  # noqa: E402
from post_processing_pipeline.helpers.utils_signalen import SignalHandler  # noqa: E402
from pyspark.sql import SparkSession  # noqa: E402
from pyspark.sql.types import StructType  # noqa: E402

from objectherkenning_openbare_ruimte.post_processing_pipeline.table_manager import (  # noqa: E402
    TableManager,
)
from objectherkenning_openbare_ruimte.settings.databricks_jobs_settings import (  # noqa: E402
    load_settings,
)


def run_update_signalen_feedback_step(
    sparkSession,
    catalog,
    schema,
    client_id,
    client_secret_name,
    access_token_url,
    base_url,
    job_process_time,
):

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
    job_process_time = datetime.fromisoformat(job_process_time)

    signalen_feedback_entries = []
    ids_of_not_updated_status = []
    for entry in signalHandler.get_pending_signalen_notifications().collect():
        id = entry["id"]
        signal_status = signalHandler.get_signal(sig_id=entry["signal_id"])["status"]
        if signal_status["state_display"] != "Gemeld":
            print(f"status is {signal_status['state_display']}")
            print(f"id is {entry['id']}")
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
            print(type(job_process_time))
            print(f"job time is: {job_process_time}")
            table_entry = (entry["signal_id"], status, text, user, formatted_date)
            signalen_feedback_entries.append(table_entry)
        else:
            ids_of_not_updated_status.append(id)

    signalen_feedback_df = signalHandler.get_signalen_feedback()
    filtered_schema = StructType(
        [
            field
            for field in signalen_feedback_df.schema.fields
            if field.name not in {"id", "processed_at"}
        ]
    )

    print(signalen_feedback_entries)
    signalen_feedback_df = spark.createDataFrame(  # noqa: F821
        signalen_feedback_entries, filtered_schema
    )
    display(signalen_feedback_df)  # noqa: F821
    signalen_feedback_df.write.mode("append").saveAsTable(
        f"{signalHandler.catalog_name}.oor.bronze_signal_notifications_feedback"
    )
    print(
        f"01: Appended {len(signalen_feedback_entries)} rows to bronze_signal_notifications_feedback."
    )

    print(ids_of_not_updated_status)
    tableManager.update_status(
        table_name="gold_signal_notifications",
        job_process_time=job_process_time,
        exclude_ids=ids_of_not_updated_status,
    )


if __name__ == "__main__":
    sparkSession = SparkSession.builder.appName("SignalenFeedback").getOrCreate()
    databricks_environment = get_databricks_environment(sparkSession)
    settings = load_settings("../../config.yml")["databricks_pipelines"][
        f"{databricks_environment}"
    ]
    run_update_signalen_feedback_step(
        sparkSession=sparkSession,
        catalog=settings["catalog"],
        schema=settings["schema"],
        client_id=settings["signalen"]["client_id"],
        client_secret_name=settings["signalen"]["client_secret_name"],
        access_token_url=settings["signalen"]["access_token_url"],
        base_url=settings["signalen"]["base_url"],
        job_process_time="2024-07-30 13:00:00",
    )
