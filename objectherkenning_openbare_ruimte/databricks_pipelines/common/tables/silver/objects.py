import pyspark.sql.functions as F
from pyspark.sql.window import Window

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(cls, send_limits={}):
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )
        filtered_df = TableManager.spark.table(table_full_name).filter(
            (F.col("status") == "Pending")
            & (F.col("score") >= 0.4)
            & (F.col("is_private_terrain") == False)
        )

        if not send_limits:
            loaded_count = filtered_df.count()
            print(
                f"Loaded {loaded_count} rows (no send_limits provided) from {table_full_name}."
            )
            return filtered_df

        # Prepare a window to rank rows within each object_class.
        window_spec = Window.partitionBy("object_class").orderBy(F.col("score").desc())
        df_with_row_number = filtered_df.withColumn(
            "row_number", F.row_number().over(window_spec)
        )

        # Build a condition that applies the appropriate limit for each object_class.
        condition = None
        for obj_class, limit in send_limits.items():
            obj_condition = (F.col("object_class") == obj_class) & (
                F.col("row_number") <= limit
            )
            condition = (
                obj_condition if condition is None else condition | obj_condition
            )

        results = df_with_row_number.filter(condition).drop("row_number")

        print(
            f"Loaded {results.count()} rows with top {send_limits} scores from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
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
