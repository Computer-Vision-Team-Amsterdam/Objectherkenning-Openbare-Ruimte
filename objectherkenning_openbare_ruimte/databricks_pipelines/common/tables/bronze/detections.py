import pyspark.sql.functions as F

from objectherkenning_openbare_ruimte.databricks_pipelines.common.tables.table_manager import (
    TableManager,
)


class BronzeDetectionMetadataManager(TableManager):
    table_name: str = "bronze_detection_metadata"

    def filter_valid_metadata(silver_frame_metadata_df):
        bronze_detection_metadata = TableManager.load_pending_rows_from_table()
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_df = silver_frame_metadata_df.alias("silver_frame")
        valid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_df,
                F.col("bronze_detection.image_name")
                == F.col("silver_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )
        print("Processed valid detection metadata.")

        return valid_metadata

    def filter_invalid_metadata(silver_frame_metadata_quarantine_df):
        bronze_detection_metadata = TableManager.load_pending_rows_from_table()
        bronze_detection_metadata = bronze_detection_metadata.alias("bronze_detection")
        silver_frame_metadata_quarantine_df = silver_frame_metadata_quarantine_df.alias(
            "quarantine_frame"
        )
        invalid_metadata = (
            bronze_detection_metadata.join(
                silver_frame_metadata_quarantine_df,
                F.col("bronze_detection.image_name")
                == F.col("quarantine_frame.image_name"),
            )
            .filter(F.col("bronze_detection.status") == "Pending")
            .select("bronze_detection.*")
        )

        print("Processed invalid detection metadata.")

        return invalid_metadata
