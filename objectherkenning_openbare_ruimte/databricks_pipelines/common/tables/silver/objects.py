import pyspark.sql.functions as F
from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "silver_objects_per_day",
    ):
        super().__init__(spark, catalog, schema, table_name)

    def get_top_pending_records(self, limit=20):
        table_full_name = f"{self.catalog}.{self.schema}.{self.table_name}"
        results = (
            self.spark.table(table_full_name)
            .filter(
                (F.col("status") == "Pending")
                & (F.col("object_class") == 2)
                & (F.col("score") >= 0.4)
            )
            .orderBy(F.col("score").desc())
            .limit(limit)
        )
        print(
            f"Loaded {results.count()} rows with top {limit} scores from {self.catalog}.{self.schema}.{self.table_name}."
        )
        return results

    def get_detection_ids_to_delete_current_run(self, job_date: str):
        return self.get_table().filter(
            (F.col("score") > 1)
            & (F.date_format(F.col("processed_at"), "yyyy-MM-dd") == job_date)
        )


class SilverObjectsPerDayQuarantineManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "silver_objects_per_day_quarantine",
    ):
        super().__init__(spark, catalog, schema, table_name)
