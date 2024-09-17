from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeSignalNotificationsFeedbackManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "bronze_signal_notifications_feedback",
    ):
        super().__init__(spark, catalog, schema, table_name)
