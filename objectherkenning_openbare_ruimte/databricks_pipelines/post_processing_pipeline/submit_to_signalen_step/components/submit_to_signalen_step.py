from typing import Any

from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common import (
    SignalHandler,
    get_job_process_time,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables import (
    GoldSignalNotificationsManager,
    SilverEnrichedDetectionMetadataManager,
    SilverEnrichedDetectionMetadataQuarantineManager,
)


class SubmitToSignalenStep:
    """
    Process pending detections according to the configured active tasks.
    """

    def __init__(
        self,
        spark_session: SparkSession,
        catalog: str,
        schema: str,
        settings: dict[str, Any],
    ):
        self.spark_session = spark_session
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
            spark_session=spark_session,
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
        """
        Check which stadsdelen are active for this run, and process them one by
        one.
        """
        active_stadsdelen = self.active_task_config.keys()

        for stadsdeel in active_stadsdelen:
            print(f"\n=== Processing detections for stadsdeel {stadsdeel} ===\n")
            self._process_stadsdeel(
                stadsdeel=stadsdeel, config=self.active_task_config[stadsdeel]
            )

    def _process_stadsdeel(self, stadsdeel: str, config: dict[str, Any]):
        """
        Process pending detections for the specified stadsdeel and create
        signals following the configured send limits.

        Parameters
        ----------
        stadsdeel: str
            Name of the stadsdeel
        config: dict[str, Any]
            Configuration for this stadsdeel in the format

            {
                "active_object_classes": [2, 3, 4]
                "send_limit": {
                    2: 4
                    3: 3
                    4: 3
                }
            }
        """
        send_limits = config.get("send_limit", {})

        top_scores_df = SilverEnrichedDetectionMetadataManager.get_top_pending_records(
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
                        fields_to_remove={
                            GoldSignalNotificationsManager.id_column,
                            "processed_at",
                        },
                    )
                )
                successful_df = self.spark_session.createDataFrame(
                    successful_notifications, schema=modified_schema
                )
                GoldSignalNotificationsManager.insert_data(df=successful_df)

            if unsuccessful_notifications:
                modified_schema = SilverEnrichedDetectionMetadataQuarantineManager.remove_fields_from_table_schema(
                    fields_to_remove={
                        GoldSignalNotificationsManager.id_column,
                        "processed_at",
                    },
                )
                unsuccessful_df = self.spark_session.createDataFrame(
                    unsuccessful_notifications, schema=modified_schema
                )
                SilverEnrichedDetectionMetadataQuarantineManager.insert_data(
                    df=unsuccessful_df
                )

        # We only want to set to "processed" the rows belonging to this stadsdeel
        processed_ids = (
            SilverEnrichedDetectionMetadataManager.get_pending_ids_for_stadsdeel(
                stadsdeel
            )
        )
        SilverEnrichedDetectionMetadataManager.update_status(
            job_process_time=self.job_process_time,
            id_column="detection_id",
            only_ids=processed_ids,
        )
