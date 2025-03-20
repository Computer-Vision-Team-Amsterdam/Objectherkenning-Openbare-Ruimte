import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(cls, privateTerrainHandler, send_limits={}):
        signals_to_send = []
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )

        for obj_class, send_limit in send_limits.items():
            print(
                f"\nProcessing object_class: '{obj_class}' (send limit: {send_limit})"
            )

            base_df = (
                TableManager.spark.table(table_full_name)
                .filter(
                    (F.col("status") == "Pending")
                    & (F.col("score") >= 0.4)
                    & (F.col("object_class") == obj_class)
                )
                .orderBy(F.col("score").desc())
            )

            # Collect candidates ordered by score to the driver.
            candidates_to_send = base_df.collect()
            signals_for_obj_class = []

            # Iterate over each candidate in order.
            for candidate in candidates_to_send:
                # Check whether this candidate is on private terrain.
                is_private, _ = privateTerrainHandler.check_private_terrain(candidate)
                if is_private:
                    print(
                        f"  Skipping detection {candidate.detection_id} because it is on private terrain."
                    )
                    continue
                else:
                    signals_for_obj_class.append(candidate)

                if len(signals_for_obj_class) >= send_limit:
                    break

            if signals_for_obj_class:
                if len(signals_for_obj_class) < send_limit:
                    print(
                        f"  Only {len(signals_for_obj_class)} non-private terrain detections found for '{obj_class}' (send limit {send_limit})."
                    )
                signals_to_send.extend(signals_for_obj_class)
            else:
                print(
                    f"  No non-private terrain candidates present for object_class: '{obj_class}'."
                )

        if signals_to_send:
            final_df = TableManager.spark.createDataFrame(signals_to_send)
            print(
                f"Loaded {final_df.count()} rows to send from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
            )
            return final_df
        else:
            print("No non-private detections found across all object classes.")
            return None

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
