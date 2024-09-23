import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @staticmethod
    def get_top_pending_records(limit=20):
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{TableManager.table_name}"
        )
        results = (
            TableManager.spark.table(table_full_name)
            .filter(
                (F.col("status") == "Pending")
                & (F.col("object_class") == 2)
                & (F.col("score") >= 0.4)
            )
            .orderBy(F.col("score").desc())
            .limit(limit)
        )
        print(
            f"Loaded {results.count()} rows with top {limit} scores from {TableManager.catalog}.{TableManager.schema}.{TableManager.table_name}."
        )
        return results

    @staticmethod
    def get_detection_ids_to_delete_current_run(job_date: str):
        return TableManager.get_table().filter(
            (F.col("score") > 1)
            & (F.date_format(F.col("processed_at"), "yyyy-MM-dd") == job_date)
        )


class SilverObjectsPerDayQuarantineManager(TableManager):
    table_name: str = "silver_objects_per_day_quarantine"
