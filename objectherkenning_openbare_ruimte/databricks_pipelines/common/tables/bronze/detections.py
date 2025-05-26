import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeDetectionMetadataManager(TableManager):
    table_name: str = "bronze_detection_metadata"
    id_column: str = "detection_id"

    @classmethod
    def filter_valid_metadata(cls, silver_frame_metadata_df):
        bronze_detection_metadata = cls.load_pending_rows_from_table()
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_df = silver_frame_metadata_df.alias("silver_frame")
        valid_metadata = bronze_detection_metadata.join(
            silver_frame_metadata_df,
            F.col("bronze_detection.frame_id") == F.col("silver_frame.frame_id"),
        ).select("bronze_detection.*")
        print("Filtered valid detection metadata.")

        return valid_metadata

    @classmethod
    def filter_invalid_metadata(cls, silver_frame_metadata_quarantine_df):
        bronze_detection_metadata = cls.load_pending_rows_from_table()
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_quarantine_df = silver_frame_metadata_quarantine_df.alias(
            "silver_frame_quarantine"
        )
        invalid_metadata = bronze_detection_metadata.join(
            silver_frame_metadata_quarantine_df,
            F.col("bronze_detection.frame_id")
            == F.col("silver_frame_quarantine.frame_id"),
        ).select("bronze_detection.*")

        print("Filtered invalid detection metadata.")

        return invalid_metadata
