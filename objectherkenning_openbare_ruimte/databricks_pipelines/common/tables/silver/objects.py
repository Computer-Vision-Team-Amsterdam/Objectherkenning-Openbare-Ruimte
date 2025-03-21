from typing import Any, List

import pyspark.sql.functions as F
from pyspark.sql import Row

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(cls, privateTerrainHandler, send_limits={}):
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )
        signals_to_send_ids = []

        print(
            f"Loading top pending detections on public terrain to send from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}..."
        )
        for obj_class, send_limit in send_limits.items():
            candidates = cls.get_pending_candidates(table_full_name, obj_class)
            public_terrain_ids = cls.get_public_terrain_ids(
                candidates, privateTerrainHandler, send_limit, obj_class
            )
            signals_to_send_ids.extend(public_terrain_ids)

        if signals_to_send_ids:
            final_df = TableManager.spark.table(table_full_name).filter(
                F.col("detection_id").isin(signals_to_send_ids)
            )
            final_count: int = final_df.count()
            print(
                f"Loaded {final_count} detections on public terrain to send from {TableManager.catalog}.{TableManager.schema}.{cls.table_name}."
            )
            return final_df
        else:
            print("No public detections found across all object classes.")
            return None

    @classmethod
    def get_pending_candidates(cls, table_full_name: str, obj_class: str) -> List[Row]:
        """Fetches candidates for a given object class that are pending and meet the score criteria."""
        base_df = (
            TableManager.spark.table(table_full_name)
            .filter(
                (F.col("status") == "Pending")
                & (F.col("score") >= 0.4)
                & (F.col("object_class") == obj_class)
            )
            .orderBy(F.col("score").desc())
        )
        return base_df.collect()

    @classmethod
    def get_public_terrain_ids(
        cls,
        candidates: List[Row],
        privateTerrainHandler: Any,
        send_limit: int,
        obj_class: str,
    ) -> List[Any]:
        """Processes candidate records, filtering out those on private terrain and returning valid detection IDs."""
        valid_ids = []
        filtered_private_count = 0

        for candidate in candidates:
            if privateTerrainHandler.check_private_terrain(candidate):
                filtered_private_count += 1
                print(
                    f"  Skipping detection {candidate.detection_id} because it is on private terrain."
                )
                continue

            valid_ids.append(candidate.detection_id)
            if len(valid_ids) >= send_limit:
                break

        if valid_ids:
            if len(valid_ids) < send_limit:
                print(
                    f"  Only {len(valid_ids)} detections on public terrain found for '{obj_class}' (send limit {send_limit})."
                )
        else:
            print(
                f"  No public terrain candidates present for object_class: '{obj_class}'."
            )
        return valid_ids

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
