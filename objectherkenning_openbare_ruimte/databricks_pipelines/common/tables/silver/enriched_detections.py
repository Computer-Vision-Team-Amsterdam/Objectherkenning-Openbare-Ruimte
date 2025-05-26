from typing import Any, Dict, List, Optional

import pyspark.sql.functions as F
from pyspark.sql import DataFrame, Row

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class SilverEnrichedDetectionMetadataManager(TableManager):
    table_name: str = "silver_enriched_detection_metadata"
    id_column: str = "detection_id"
    score_threshold = 0.4

    @classmethod
    def get_top_pending_records(
        cls,
        exclude_private_terrain_detections: bool = False,
        stadsdeel: Optional[str] = None,
        active_object_classes: List[int] = [],
        send_limits: Dict[int, int] = {},
    ) -> Optional[DataFrame]:
        """
        Collects and returns top pending records for each active object class within send limits.

        Parameters:
            exclude_private_terrain_detections (bool): Flag indicating whether to exclude detections on private terrain.
            az_tenant_id (str): Azure tenant ID required for database access.
            db_host (str): Host address of the database.
            db_name (str): Name of the database.
            stadsdeel (str): Name of stadsdeel or None to get all results
            active_object_classes (List[int]): list of object classes to filter by
            send_limits: A dictionary mapping each object class to its maximum number of records to send.

        Returns:
            A DataFrame containing valid pending detections if available, otherwise None.
        """
        table_full_name = (
            f"{TableManager.catalog}.{TableManager.schema}.{cls.table_name}"
        )
        detection_ids_to_send = []

        print(f"Loading top pending detections to send from {table_full_name} ...")

        # Retrieve all distinct object classes that have pending candidates.
        if stadsdeel:
            pending_obj_classes = (
                TableManager.spark_session.table(table_full_name)
                .filter(
                    (F.col("status") == "Pending")
                    & (F.lower(F.col("stadsdeel")) == stadsdeel.lower())
                    & (F.col("score") >= cls.score_threshold)
                )
                .select("object_class")
                .distinct()
                .rdd.flatMap(lambda x: x)
                .collect()
            )
        else:
            pending_obj_classes = (
                TableManager.spark_session.table(table_full_name)
                .filter(
                    (F.col("status") == "Pending")
                    & (F.col("score") >= cls.score_threshold)
                )
                .select("object_class")
                .distinct()
                .rdd.flatMap(lambda x: x)
                .collect()
            )
        if active_object_classes:
            pending_obj_classes = set(pending_obj_classes).intersection(
                active_object_classes
            )

        for obj_class in pending_obj_classes:
            send_limit = send_limits.get(obj_class, None)
            candidate_rows = cls.get_pending_candidates(
                table_full_name,
                obj_class,
                send_limit,
                exclude_private_terrain_detections,
                stadsdeel,
            )
            print(
                f"  Found {len(candidate_rows)} detections for object class '{obj_class}' (send limit {send_limit if send_limit else 'not set'})."
            )
            detection_ids = [candidate[cls.id_column] for candidate in candidate_rows]
            detection_ids_to_send.extend(detection_ids)

        if detection_ids_to_send:
            detections_to_send_df = TableManager.spark_session.table(
                table_full_name
            ).filter(F.col(cls.id_column).isin(detection_ids_to_send))
            print(
                f"Loaded {detections_to_send_df.count()} valid detections to send from {table_full_name}."
            )
            return detections_to_send_df
        else:
            print("No valid detections to send found across all object classes.")
            return None

    @classmethod
    def get_pending_candidates(
        cls,
        table_full_name: str,
        obj_class: int,
        send_limit: Optional[int] = None,
        exclude_private_terrain_detections: bool = False,
        stadsdeel: Optional[str] = None,
    ) -> List[Row]:
        """
        Fetches candidate detections for a given object class that are pending and meet the score criteria.

        Parameters:
            table_full_name: The fully-qualified table name.
            obj_class: The object class for which pending detections should be fetched.
            exclude_private_terrain_detections: Whether to exclude detections on private terrain.
            stadsdeel (str | None): Name of stadsdeel or None to get all results

        Returns:
            A list of Rows representing candidate detections.
        """
        pending_candidates_df: DataFrame = (
            TableManager.spark_session.table(table_full_name)
            .filter(
                (F.col("status") == "Pending")
                & (F.col("score") >= cls.score_threshold)
                & (F.col("object_class") == obj_class)
            )
            .orderBy(F.col("score").desc())
        )
        if exclude_private_terrain_detections:
            pending_candidates_df = pending_candidates_df.filter(
                F.col("private_terrain") == False
            )
        if stadsdeel:
            pending_candidates_df = pending_candidates_df.filter(
                F.lower(F.col("stadsdeel")) == stadsdeel.lower()
            )
        if send_limit is not None:
            return pending_candidates_df.take(send_limit)
        else:
            return pending_candidates_df.collect()

    @classmethod
    def get_detection_ids_to_keep_current_run(cls, job_date: str) -> List[Any]:
        """
        Retrieves detection IDs to keep for the current run by filtering records
        processed on the given job_date and with a score above a certain
        threshold.

        Parameters:
            job_date: The date (in "yyyy-MM-dd" format) to filter the
            processed_at field.

        Returns:
            A list of detection IDs that match the filtering criteria.
        """
        return (
            cls.get_table()
            .filter(
                (F.col("score") >= cls.score_threshold)
                & (F.date_format(F.col("processed_at"), "yyyy-MM-dd") == job_date)
            )
            .select(cls.id_column)
            .rdd.flatMap(lambda x: x)
            .collect()
        )

    @classmethod
    def get_detection_ids_candidates_for_deletion(cls, job_date: str) -> List[Any]:
        """
        Retrieves detection IDs that are candidates for deletion for the current
        run by filtering records processed on the given job_date and with a
        score below a certain threshold.

        Parameters:
            job_date: The date (in "yyyy-MM-dd" format) to filter the
            processed_at field.

        Returns:
            A list of detection IDs that match the filtering criteria.
        """
        return (
            cls.get_table()
            .filter(
                (F.col("score") < cls.score_threshold)
                & (F.date_format(F.col("processed_at"), "yyyy-MM-dd") == job_date)
            )
            .select(cls.id_column)
            .rdd.flatMap(lambda x: x)
            .collect()
        )

    @classmethod
    def get_pending_ids_for_stadsdeel(cls, stadsdeel: str) -> List[int]:
        """
        Retrieves detection IDs by filtering pending records by stadsdeel name.

        Parameters:
            stadsdeel: the stadsdeel

        Returns:
            A list of detection IDs that match the filtering criteria.
        """
        return (
            cls.get_table()
            .filter(
                (F.col("status") == "Pending")
                & (F.lower(F.col("stadsdeel")) == stadsdeel.lower())
            )
            .select(cls.id_column)
            .rdd.flatMap(lambda x: x)
            .collect()
        )


class SilverEnrichedDetectionMetadataQuarantineManager(TableManager):
    table_name: str = "silver_enriched_detection_metadata_quarantine"
    id_column: str = "detection_id"
