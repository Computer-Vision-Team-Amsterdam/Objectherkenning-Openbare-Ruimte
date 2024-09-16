from pyspark.sql import SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDay(TableManager):
    def __init__(self, spark: SparkSession, catalog: str, schema: str):
        # Call the constructor of the base class
        super().__init__(spark, catalog, schema)
        # Define the specific table name for this subclass
        self.table_name = "silver_objects_per_day"

    def get_top_pending_records(self, limit=20):
        # Select all rows where status is 'Pending' and detections are containers, sort by score in descending order, and limit the results to the top 10
        select_query = f"""
        SELECT * FROM {self.catalog}.{self.schema}.{self.table_name}
        WHERE status = 'Pending' AND object_class = 2 AND score >= 0.4
        ORDER BY score DESC
        LIMIT {limit}
        """  # nosec
        results = self.spark.sql(select_query)
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
        return results
