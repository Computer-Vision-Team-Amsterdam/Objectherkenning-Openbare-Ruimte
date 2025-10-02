import datetime
from typing import Dict, List, Optional

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
    def get_top_pending_records_for_stadsdeel(
        cls,
        stadsdeel: str,
        exclude_private_terrain_detections: bool = False,
        active_object_classes: List[int] = [],
        send_limits: Dict[int, int] = {},
        score_threshold: float = score_threshold,
        skip_ids: list[int] = [],
        detection_date: Optional[datetime.date] = None,
    ) -> Optional[DataFrame]:
        """
        Collects and returns top pending records for each active object class within send limits.

        Parameters:
            stadsdeel (str): Name of stadsdeel
            exclude_private_terrain_detections (bool): Flag indicating whether to exclude detections on private terrain.
            active_object_classes (List[int]): list of object classes to filter by
            send_limits: A dictionary mapping each object class to its maximum number of records to send.
            score_threshold: the minimum score required for records to be returned
            skip_ids (optional): list of detection_ids to skip
            detection_date (optional): only return detections from this datetime.date

        Returns:
            A DataFrame containing valid pending detections if available, otherwise None.
        """
        detection_ids_to_send = []

        print(
            f"Loading top pending detections to send from {cls.get_table_full_name()} ..."
        )

        # Retrieve all distinct object classes that have pending candidates.
        pending_obj_classes = (
            cls.get_table()
            .filter(
                (F.col("status") == "Pending")
                & (F.lower(F.col("stadsdeel")) == stadsdeel.lower())
                & (F.col("score") >= score_threshold)
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
            candidate_rows = cls._get_pending_candidates(
                obj_class,
                send_limit,
                exclude_private_terrain_detections,
                stadsdeel,
                score_threshold,
                skip_ids,
                detection_date,
            )
            print(
                f"  Found {len(candidate_rows)} detections for object class '{obj_class}' "
                f"(send limit {send_limit if send_limit is not None else 'not set'})."
            )
            detection_ids = [candidate[cls.id_column] for candidate in candidate_rows]
            detection_ids_to_send.extend(detection_ids)

        if detection_ids_to_send:
            detections_to_send_df = cls.get_table().filter(
                F.col(cls.id_column).isin(detection_ids_to_send)
            )
            print(f"Loaded {detections_to_send_df.count()} valid detections to send.")
            return detections_to_send_df
        else:
            print("No valid detections to send found across all object classes.")
            return None

    @classmethod
    def _get_pending_candidates(
        cls,
        obj_class: int,
        send_limit: Optional[int] = None,
        exclude_private_terrain_detections: bool = False,
        stadsdeel: Optional[str] = None,
        score_threshold: float = score_threshold,
        skip_ids: list[int] = [],
        detection_date: Optional[datetime.date] = None,
    ) -> List[Row]:
        """
        Fetches candidate detections for a given object class that are pending and meet the score criteria.

        Parameters:
            obj_class: The object class for which pending detections should be fetched.
            exclude_private_terrain_detections: Whether to exclude detections on private terrain.
            stadsdeel (str | None): Name of stadsdeel or None to get all results
            score_threshold: the minimum score required for records to be returned
            skip_ids (optional): list of detection_ids to skip
            detection_date (optional): only return detections from this datetime.date

        Returns:
            A list of Rows representing candidate detections.
        """
        pending_candidates_df: DataFrame = (
            cls.get_table()
            .filter(
                (F.col("status") == "Pending")
                & (F.col("score") >= score_threshold)
                & (F.col("object_class") == obj_class)
            )
            .orderBy(F.col("score").desc())
        )
        if exclude_private_terrain_detections:
            pending_candidates_df = pending_candidates_df.filter(
                F.col("private_terrain") == False
            )
        if stadsdeel is not None:
            pending_candidates_df = pending_candidates_df.filter(
                F.lower(F.col("stadsdeel")) == stadsdeel.lower()
            )
        if detection_date is not None:
            pending_candidates_df = pending_candidates_df.filter(
                F.col("detection_date") == detection_date
            )
        if len(skip_ids) > 0:
            pending_candidates_df = pending_candidates_df.filter(
                ~F.col("detection_id").isin(skip_ids)
            )
        if send_limit is not None:
            return pending_candidates_df.take(send_limit)
        else:
            return pending_candidates_df.collect()

    @classmethod
    def get_pending_ids_for_stadsdeel(
        cls, stadsdeel: str, detection_date: Optional[datetime.date] = None
    ) -> List[int]:
        """
        Retrieves detection IDs by filtering pending records by stadsdeel name.

        Parameters:
            stadsdeel: the stadsdeel
            detection_date (optional): the detection date

        Returns:
            A list of detection IDs that match the filtering criteria.
        """
        pending_table = cls.get_table().filter(
            (F.col("status") == "Pending")
            & (F.lower(F.col("stadsdeel")) == stadsdeel.lower())
        )
        if detection_date is not None:
            pending_table = pending_table.filter(
                F.col("detection_date") == detection_date
            )
        return pending_table.select(cls.id_column).rdd.flatMap(lambda x: x).collect()


class SilverEnrichedDetectionMetadataQuarantineManager(TableManager):
    table_name: str = "silver_enriched_detection_metadata_quarantine"
    id_column: str = "detection_id"
