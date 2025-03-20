import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(cls, privateTerrainHandler, send_limits={}):
        signals_to_send_ids = []
        table_full_name = f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"

        for obj_class, send_limit in send_limits.items():
            base_df = (
                TableManager.spark.table(table_full_name)
                .filter(
                    (F.col("status") == "Pending") &
                    (F.col("score") >= 0.4) &
                    (F.col("object_class") == obj_class)
                )
                .orderBy(F.col("score").desc())
            )

            # Collect candidates ordered by score to the driver.
            candidates_to_send = base_df.collect()
            valid_ids_for_obj_class = []
            filtered_private_count = 0

            # Iterate over each candidate in order.
            for candidate in candidates_to_send:
                # Check whether this candidate is on private terrain.
                on_private_terrain = privateTerrainHandler.check_private_terrain(candidate)
                if on_private_terrain:
                    filtered_private_count += 1
                    print(f"  Skipping detection {candidate.detection_id} because it is on private terrain.")
                    continue
                else:
                    valid_ids_for_obj_class.append(candidate.detection_id)

                if len(valid_ids_for_obj_class) >= send_limit:
                    break

            print(
                f"Filtered out {filtered_private_count} detections for object_class '{obj_class}' due to private terrain."
            )
            if valid_ids_for_obj_class:
                if len(valid_ids_for_obj_class) < send_limit:
                    print(
                        f"  Only {len(valid_ids_for_obj_class)} detections on public terrain found for '{obj_class}' (send limit {send_limit})."
                    )
                signals_to_send_ids.extend(valid_ids_for_obj_class)
            else:
                print(f"  No public terrain candidates present for object_class: '{obj_class}'.")

        if signals_to_send_ids:
            # Filter the original table using the collected detection_ids.
            final_df = TableManager.spark.table(table_full_name).filter(F.col("detection_id").isin(signals_to_send_ids))
            final_count = final_df.count()
            print(
                f"Loaded {final_count} rows to send from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
            )
            return final_df
        else:
            print("No public detections found across all object classes.")
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
