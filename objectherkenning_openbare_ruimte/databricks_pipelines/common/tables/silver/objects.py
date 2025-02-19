import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(cls, limit=20):
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
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
            f"Loaded {results.count()} rows with top {limit} scores from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
        )
        return results

    @classmethod
    def get_detection_ids_to_keep_current_run(cls, job_date: str):
        return (
            cls.get_table()
            .filter(
                (F.col("score") >= 1)
                & (F.date_format(F.col("processed_at"), "yyyy-MM-dd") == job_date)
            )
            .select("detection_id")
            .rdd.flatMap(lambda x: x)
            .collect()
        )


class SilverObjectsPerDayQuarantineManager(TableManager):
    table_name: str = "silver_objects_per_day_quarantine"
