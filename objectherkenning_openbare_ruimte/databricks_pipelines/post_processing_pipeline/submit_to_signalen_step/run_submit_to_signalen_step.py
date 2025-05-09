# this fixes the caching issues, reimports all modules
dbutils.library.restartPython()  # type: ignore[name-defined] # noqa: F821

import os  # noqa: E402
from typing import Any  # noqa: E402

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


class SubmitToSignalenStep:

    def __init__(
        self,
        sparkSession: SparkSession,
        catalog: str,
        schema: str,
        settings: dict[str, Any],
    ):
        self.sparkSession = sparkSession
        self.catalog = catalog
        self.schema = schema
        self.job_process_time = get_job_process_time(
            is_first_pipeline_step=False,
        )
        self.az_tenant_id = settings["azure_tenant_id"]
        self.db_host = settings["reference_database"]["host"]
        self.db_name = settings["reference_database"]["name"]
        self.exclude_private_terrain_detections = settings["job_config"][
            "exclude_private_terrain_detections"
        ]
        self.annotate_detection_images = settings["job_config"][
            "annotate_detection_images"
        ]
        self.active_task_config = settings["job_config"]["active_task"]

        self.signalHandler = SignalHandler(
            sparkSession=sparkSession,
            catalog=catalog,
            schema=schema,
            device_id=settings["device_id"],
            signalen_settings=settings["signalen"],
            az_tenant_id=self.az_tenant_id,
            db_host=self.db_host,
            db_name=self.db_name,
            object_classes=settings["job_config"]["object_classes"]["names"],
        )

    def run_submit_to_signalen_step(self):
        active_stadsdelen = self.active_task_config.keys()

        for stadsdeel in active_stadsdelen:
            self._process_stadsdeel(
                stadsdeel=stadsdeel, config=self.active_task_config[stadsdeel]
            )

    def _process_stadsdeel(self, stadsdeel: str, config: dict[str, Any]):
        send_limits = config.get("send_limit", {})

        top_scores_df = SilverObjectsPerDayManager.get_top_pending_records(
            self.exclude_private_terrain_detections,
            self.az_tenant_id,
            self.db_host,
            self.db_name,
            stadsdeel=stadsdeel,
            active_object_classes=config.get("active_object_classes", []),
            send_limits=send_limits,
        )

        if (not top_scores_df) or top_scores_df.count() == 0:
            print("No data found for creating notifications. Stopping execution.")
        else:
            successful_notifications, unsuccessful_notifications = (
                self.signalHandler.process_notifications(
                    top_scores_df, self.annotate_detection_images
                )
            )

            if successful_notifications:
                modified_schema = (
                    GoldSignalNotificationsManager.remove_fields_from_table_schema(
                        fields_to_remove={"id", "processed_at"},
                    )
                )
                successful_df = self.sparkSession.createDataFrame(
                    successful_notifications, schema=modified_schema
                )
                GoldSignalNotificationsManager.insert_data(df=successful_df)

            if unsuccessful_notifications:
                modified_schema = SilverObjectsPerDayQuarantineManager.remove_fields_from_table_schema(
                    fields_to_remove={"id", "processed_at"},
                )
                unsuccessful_df = self.sparkSession.createDataFrame(
                    unsuccessful_notifications, schema=modified_schema
                )
                SilverObjectsPerDayQuarantineManager.insert_data(df=unsuccessful_df)

        processed_ids = SilverObjectsPerDayManager.get_pending_ids_for_stadsdeel(
            stadsdeel
        )
        SilverObjectsPerDayManager.update_status(
            job_process_time=self.job_process_time,
            id_column="detection_id",
            only_ids=processed_ids,
        )


def main():
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
    setup_tables(spark=sparkSession, catalog=catalog, schema=schema)

    submitToSignalenStep = SubmitToSignalenStep(
        sparkSession=sparkSession, catalog=catalog, schema=schema, settings=settings
    )
    submitToSignalenStep.run_submit_to_signalen_step()


if __name__ == "__main__":
    main()
