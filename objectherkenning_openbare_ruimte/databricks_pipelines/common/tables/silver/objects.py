from typing import Any, Dict, List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Row, SparkSession

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)
from objectherkenning_openbare_ruimte.databricks_pipelines.post_processing_pipeline.submit_to_signalen_step.components.private_terrain_handler import (  # noqa: E402
    PrivateTerrainHandler,
)


class SilverObjectsPerDayManager(TableManager):
    table_name: str = "silver_objects_per_day"

    @classmethod
    def get_top_pending_records(
        cls,
        exclude_private_terrain_detections: bool,
        sparkSession: SparkSession,
        az_tenant_id: str,
        db_host: str,
        db_name: str,
        send_limits: Dict[int, int] = {},
    ) -> Optional[DataFrame]:
        """
        Collects and returns top pending records for each active object class within send limits.

        Parameters:
            privateTerrainHandler: An object that provides the on_private_terrain method to check if a candidate is on private terrain.
            send_limits: A dictionary mapping each object class to its maximum number of records to send.

        Returns:
            A DataFrame containing valid pending detections if available, otherwise None.
        """
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )
        detections_to_send_ids = []

        if exclude_private_terrain_detections:
            privateTerrainHandler = PrivateTerrainHandler(
                spark=sparkSession,
                az_tenant_id=az_tenant_id,
                db_host=db_host,
                db_name=db_name,
                db_port=5432,
            )
        else:
            privateTerrainHandler = None

        print(f"Loading top pending detections to send from {table_full_name} ...")

        # Retrieve all distinct object classes that have pending candidates.
        pending_obj_classes = (
            TableManager.spark.table(table_full_name)
            .filter((F.col("status") == "Pending") & (F.col("score") >= 0.4))
            .select("object_class")
            .distinct()
            .rdd.flatMap(lambda x: x)
            .collect()
        )

        for obj_class in pending_obj_classes:
            send_limit = send_limits.get(obj_class, None)
            candidates = cls.get_pending_candidates(table_full_name, obj_class)
            valid_detection_ids = cls.get_valid_detection_ids(
                candidates, privateTerrainHandler, send_limit, obj_class
            )
            detections_to_send_ids.extend(valid_detection_ids)

        if detections_to_send_ids:
            detections_to_send_df = TableManager.spark.table(table_full_name).filter(
                F.col("detection_id").isin(detections_to_send_ids)
            )
            print(
                f"Loaded {detections_to_send_df.count()} valid detections to send from {table_full_name}."
            )
            return detections_to_send_df
        else:
            print("No valid detections to send found across all object classes.")
            return None

    @classmethod
    def get_pending_candidates(cls, table_full_name: str, obj_class: int) -> List[Row]:
        """
        Fetches candidate detections for a given object class that are pending and meet the score criteria.

        Parameters:
            table_full_name: The fully-qualified table name.
            obj_class: The object class for which pending detections should be fetched.

        Returns:
            A list of Rows representing candidate detections.
        """
        pending_candidates_df = (
            TableManager.spark.table(table_full_name)
            .filter(
                (F.col("status") == "Pending")
                & (F.col("score") >= 0.4)
                & (F.col("object_class") == obj_class)
            )
            .orderBy(F.col("score").desc())
        )
        return pending_candidates_df.collect()

    @classmethod
    def get_valid_detection_ids(
        cls,
        candidates: List[Row],
        privateTerrainHandler: Optional[PrivateTerrainHandler],
        send_limit: int,
        obj_class: int,
    ) -> List[Any]:
        """
        Processes candidate records, filtering out those on private terrain if applicable,
        and returns valid detection IDs.

        Parameters:
            candidates: List of candidate Rows to evaluate.
            privateTerrainHandler: An object to determine if a candidate is on private terrain.
            send_limit: Maximum number of detections to return.
            obj_class: The object class for which candidates are evaluated.

        Returns:
            A list of valid detection IDs from the candidate records.
        """
        valid_ids = []
        filtered_private_count = 0

        for candidate in candidates:
            # Check if the candidate is not on private terrain or if no terrain handler is provided.
            if (
                privateTerrainHandler is None
                or not privateTerrainHandler.on_private_terrain(candidate)
            ):
                valid_ids.append(candidate.detection_id)
                # If a send limit is specified, break once it is reached.
                if send_limit is not None and len(valid_ids) >= send_limit:
                    break
            else:
                filtered_private_count += 1
                print(
                    f"  Skipping detection {candidate.detection_id} because it is on private terrain."
                )
                continue

        if valid_ids:
            if send_limit is not None and len(valid_ids) < send_limit:
                print(
                    f"  Only {len(valid_ids)} detections found for '{obj_class}' (send limit {send_limit})."
                )
        else:
            print(f"  No candidates present for object_class: '{obj_class}'.")
        return valid_ids

    @classmethod
    def get_detection_ids_to_keep_current_run(cls, job_date: str) -> List[Any]:
        """
        Retrieves detection IDs to keep for the current run by filtering records processed on the given job_date
        and with a score above a certain threshold.

        Parameters:
            job_date: The date (in "yyyy-MM-dd" format) to filter the processed_at field.

        Returns:
            A list of detection IDs that match the filtering criteria.
        """
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
