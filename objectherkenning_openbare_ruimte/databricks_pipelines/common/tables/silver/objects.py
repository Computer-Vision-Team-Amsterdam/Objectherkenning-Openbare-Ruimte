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
        # Select all rows where status is 'Pending' and detections are containers, sort by score in descending order, and limit the results to the top 10
        select_query = f"""
        SELECT * FROM {self.catalog}.{self.schema}.{self.table_name}
        WHERE status = 'Pending' AND object_class = 2 AND score >= 0.4
        ORDER BY score DESC
        LIMIT {limit}
        """  # nosec
        results = self.spark.sql(select_query)
        print(
            f"Loaded {results.count()} rows with top {limit} scores from {self.catalog}.{self.schema}.{self.table_name}."
        )
        return results

    def get_top_pending_records_no_sql(self, limit=20):
        table_full_name = f"{self.catalog}.{self.schema}.{self.table_name}"
        results = (
            self.spark.table(table_full_name)
            .filter(
                (self.spark.col("status") == "Pending")
                & (self.spark.col("object_class") == 2)
                & (self.spark.col("score") >= 0.4)
            )
            .orderBy(self.spark.col("score").desc())
            .limit(limit)
        )
        print(
            f"Loaded {results.count()} rows with top {limit} scores from {self.catalog}.{self.schema}.{self.table_name}."
        )
        return results


class SilverObjectsPerDayQuarantineManager(TableManager):
    def __init__(
        self,
        spark: SparkSession,
        catalog: str,
        schema: str,
        table_name: str = "silver_objects_per_day_quarantine",
    ):
        super().__init__(spark, catalog, schema, table_name)
